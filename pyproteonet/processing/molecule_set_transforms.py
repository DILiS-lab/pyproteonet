from typing import Optional, Union
from io import StringIO
import requests

import pandas as pd
from tqdm.auto import tqdm
from pyopenms import ProteaseDigestion, AASequence

from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


def map_protein_sequence(
    input: MoleculeSet,
    uniprot_id_column: Optional[str] = None,
    result_column: Optional[str] = "sequence",
    request_size: int = 50,
) -> pd.Series:
    """
    Retrieve uniprot sequences based on a list of uniprot sequence identifier.
    For large lists it is recommended to perform batch retrieval.
    Code based on https://www.biostars.org/p/94422/ and https://www.biostars.org/p/67822/

    Args:
        input (Union[Dataset, MoleculeSet]): MoleculeSet to retrieve protein ids from
        uniprot_id_column (str, optional): Protein molecule column to retrieve uniprot ids. If None, the protein index is used.
          Defaults to None.
        output_column (str): Column in protein Dataframe to write sequences to.
        inplace (bool): Whether to copy the MoleculeSet/Dataset before retrieving the sequences.
        request_size (int, optional): Number of proteins to batch in one request. Defaults to 50.

    Returns:
        Optional[MoleculeSet]: The transformed input or None if inplace=True
    """
    molecules = input.molecules
    if "protein" not in molecules:
        raise KeyError('MoleculeSet must have "protein" to map protein sequences.')
    molecules = molecules["protein"]
    uniprot_ids = molecules[uniprot_id_column] if uniprot_id_column is not None else molecules.index
    base_url = "http://rest.uniprot.org/uniprotkb/search"
    start = 0
    results = []
    for start in tqdm(range(0, len(uniprot_ids), request_size)):
        params = {
            #'from': "ACC",
            #'to': 'ACC',
            "format": "tsv",
            "query": " OR ".join([f"accession:{id}" for id in uniprot_ids[start : start + request_size]]),
            #'query': " OR ".join(uniprot_ids[start:start+page_size]),
            "fields": "accession,sequence",
            #'size': min(request_size, len(uniprot_ids) - start),
        }
        # data = data.encode('ascii')
        response = requests.get(base_url, params=params)
        df_result = pd.read_csv(StringIO(response.content.decode("utf-8")), sep="\t")
        df_result.columns = ["entry", "sequence"]
        results.append(df_result)
        while "Link" in response.headers:
            link = response.headers["Link"].partition(">")[0][1:]
            response = requests.get(link)
            df_result = pd.read_csv(StringIO(response.content.decode("utf-8")), sep="\t")
            df_result.columns = ["entry", "sequence"]
            results.append(df_result)
    results = pd.concat(results, ignore_index=True)
    results.set_index("entry", drop=True, inplace=True, verify_integrity=True)
    assert len(results) == len(uniprot_ids) and uniprot_ids.isin(results.index).all()
    results.loc[results.sequence.isna(), "sequence"] = ""
    if result_column is not None:
        molecules[result_column] = results.loc[uniprot_ids, "sequence"].values
        return molecules['sequence']
    else:
        return results['sequence']

def num_theoretical_peptides(
    molecule_set: MoleculeSet,
    sequence_column: str = "sequence",
    enzyme: str = "Trypsin",
    min_peptide_length: int = 7,
    max_peptide_length: int = 30,
    result_column: Optional[str] = "num_theoretical_peptides",
) -> pd.Series:
    if "protein" not in molecule_set.molecules:
        raise KeyError("The MoleculeSet must contain proteins!")
    sequences = molecule_set.molecules["protein"][sequence_column]
    digestor = ProteaseDigestion()
    digestor.setEnzyme(enzyme)
    num_peps = []
    for sequence in sequences:
        res = []
        digestor.digest(AASequence().fromString(sequence), res, min_peptide_length, max_peptide_length)
        num_peps.append(len(set(res)))
    if result_column is not None:
        molecule_set.molecules['protein'][result_column] = num_peps
        return molecule_set.molecules['protein'][result_column]
    else:
        return pd.Series(num_peps, index=sequences.index)

