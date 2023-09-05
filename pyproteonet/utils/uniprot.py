from typing import Optional, Union, List
from io import StringIO
import requests

import pandas as pd
from tqdm.auto import tqdm
from pyopenms import ProteaseDigestion, AASequence

from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry


POLLING_INTERVAL = 3
API_URL = "https://rest.uniprot.org"

class UniprotMapper:

    def __init__(self):
        self.retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))


    def check_response(self, response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())


    def submit_id_mapping(self, from_db, to_db, ids):
        request = requests.post(
            f"{API_URL}/idmapping/run",
            data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        )
        self.check_response(request)
        #print(request.json())
        return request.json()["jobId"]


    def get_next_link(self, headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)


    def check_id_mapping_results_ready(self, job_id, verbose: bool = False):
        while True:
            request = self.session.get(f"{API_URL}/idmapping/status/{job_id}")
            self.check_response(request)
            j = request.json()
            if verbose:
                print(j)
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    if verbose:
                        print(f"Retrying in {POLLING_INTERVAL}s")
                    time.sleep(POLLING_INTERVAL)
                else:
                    raise Exception(j["jobStatus"])
            else:
                return bool(j["results"] or j["failedIds"])


    def get_batch(self, batch_response, file_format, compressed):
        batch_url = self.get_next_link(batch_response.headers)
        while batch_url:
            batch_response = self.session.get(batch_url)
            batch_response.raise_for_status()
            yield self.decode_results(batch_response, file_format, compressed)
            batch_url = self.get_next_link(batch_response.headers)


    def combine_batches(self, all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results


    def get_id_mapping_results_link(self, job_id):
        url = f"{API_URL}/idmapping/details/{job_id}"
        request = self.session.get(url)
        self.check_response(request)
        #print(request.json())
        return request.json()["redirectURL"]


    def decode_results(self, response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text


    def get_xml_namespace(self, element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""


    def merge_xml_results(self, xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", self.get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)


    def print_progress_batches(self, batch_index, size, total):
        n_fetched = min((batch_index + 1) * size, total)
        print(f"Fetched: {n_fetched} / {total}")


    def get_id_mapping_results_search(self, url, verbose: bool = False):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = query["compressed"][0].lower() == "true" if "compressed" in query else False
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = self.session.get(url)
        self.check_response(request)
        results = self.decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        if verbose:
            self.print_progress_batches(0, size, total)
        for i, batch in enumerate(self.get_batch(request, file_format, compressed), 1):
            results = self.combine_batches(results, batch, file_format)
            if verbose:
                self.print_progress_batches(i, size, total)
        if file_format == "xml":
            return self.merge_xml_results(results)
        return results


    def get_id_mapping_results_stream(self, url):
        if "/stream/" not in url:
            url = url.replace("/results/", "/results/stream/")
        request = self.session.get(url)
        self.check_response(request)
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        compressed = query["compressed"][0].lower() == "true" if "compressed" in query else False
        return self.decode_results(request, file_format, compressed)


    def map_uniprot(self, ids: pd.Series, to_db: str, from_db: str = "UniProtKB_AC-ID", verbose: bool = False):
        ids = ids.drop_duplicates()
        ids_t = (pd.Series(ids.index.values, index=ids))
        job_id = self.submit_id_mapping(from_db=from_db, to_db=to_db, ids=list(ids_t.index))
        if self.check_id_mapping_results_ready(job_id, verbose=verbose):
            link = self.get_id_mapping_results_link(job_id)
            results = self.get_id_mapping_results_search(link, verbose=verbose)
        # Equivalently using the stream endpoint which is more demanding
        # on the API and so is less stable:
        # results = get_id_mapping_results_stream(link)
        results = pd.DataFrame(results['results'])
        results.loc[:, 'from']=results['from'].map(ids_t)
        results.set_index('from', drop=True, inplace=True)
        results.index.set_names(ids.index.names, inplace=True)
        return results['to']
