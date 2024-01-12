from pathlib import Path
import urllib
import zipfile
import tarfile

import pandas as pd

from ..data.dataset import Dataset
from .maxquant import load_maxquant

def load_maxlfq_benchmark(path: Path)->Dataset:
    """If path does not exist, download and extract the MaxLFQ benchmark dataset (PXD000279) from PRIDE.

    Args:
        path (Path): path to either load dataset from or download dataset to

    Returns:
        Dataset: The loaded dataset
    """
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

def load_crohns_disease(path: Path)->Dataset:
    """If path does not exist, download and extract a Crohn's disease dataset (PXD000561) from PRIDE.

    Args:
        path (Path): path to either load dataset from or download dataset to.

    Returns:
        Dataset: The loaded dataset
    """
    if not path.exists():
        path.mkdir(parents=True)
        url = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2016/09/PXD002882/MaxQuantOutput.tar.gz"
        zip_path = path / 'PXD000561_MaxQuantOutput.tar.gz'
        urllib.request.urlretrieve(url, zip_path)
        with tarfile.TarFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    df = pd.read_csv(path / 'peptides.txt', sep='\t')
    samples = [c.replace('Intensity ', '') for c in df.columns if 'Intensity' in c]
    samples = [c for c in samples if not c.startswith(('L ', 'H '))][3:]
    mask = ~(df[['Intensity ' + s for s in samples]] == 0).all(axis=1)
    df = df[mask]
    crohns_ds = load_maxquant(peptides_table=df, samples=samples)
    return crohns_ds