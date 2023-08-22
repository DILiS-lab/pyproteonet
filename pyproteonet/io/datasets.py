from pathlib import Path
import urllib
import zipfile

from ..data.dataset import Dataset
from .maxquant import load_maxquant

def load_maxlfq_benchmark(path: Path)->Dataset:
    if not path.exists():
        path.mkdir(parents=True)
        url = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2014/09/PXD000279/proteomebenchmark.zip"
        zip_path = path / 'proteomebenchmark.zip'
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    ds = load_maxquant(peptides_table=path/'peptides.txt', protein_groups_table=path/'proteinGroups.txt',
                       samples=['L1','L2','L3','H1','H2','H3'],
                      )
    return ds