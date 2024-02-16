from typing import Optional
from pathlib import Path
import urllib
import zipfile
import tarfile

import pandas as pd

from ..data.dataset import Dataset
from .maxquant import load_maxquant

def load_example_dataset(path: Optional[Path] = None)->Dataset:
    """A small exmple dataset from PRIDE (PXD028038). If path does not exist, download and extract the dataset from PRIDE.

    Args:
        path (Optional[Path]): path to either load dataset from or download dataset to, if not given the home directory is used.

    Returns:
        Dataset: The loaded dataset
    """
    if path is None:
        path = Path.home() / 'pyproteonet' / 'datasets' / 'PXD028038'
    if not path.exists():
        path.mkdir(parents=True)
        url = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2022/04/PXD028038/HER2_MD1_output.zip"
        zip_path = path / 'HER2_MD1_output.zip'
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
        
    ds = load_maxquant(peptides_table=path/'HER2_MD1_output'/'peptides_quant.txt', 
                    protein_groups_table=path /'HER2_MD1_output'/'proteinGroups_quant.txt', peptide_columns=['sequence'],
                    peptide_value_columns=['intensity'], protein_group_value_columns=['intensity'],
                    protein_group_columns=['protein ids'], peptide_protein_group_map_column='protein group ids', remove_invalid_mappings=True)
    return ds

def load_maxlfq_benchmark_dataset(path: Optional[Path] = None)->Dataset:
    """MaxLFQ benchmark dataset (PRIDE id: PXD000279). If path does not exist, download and extract the

    Args:
        path (Optional[Path]): path to either load dataset from or download dataset to

    Returns:
        Dataset: The loaded dataset
    """
    if path is None:
        path = Path.home() / 'pyproteonet' / 'datasets' / 'PXD000279'
    if not path.exists():
        path.mkdir(parents=True)
        url = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2014/09/PXD000279/proteomebenchmark.zip"
        zip_path = path / 'proteomebenchmark.zip'
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    ds = load_maxquant(peptides_table=path/'peptides.txt', protein_groups_table=path/'proteinGroups.txt',
                       protein_group_value_columns=['LFQ intensity'],
                       samples=['L1','L2','L3','H1','H2','H3'], protein_group_columns=['Fasta headers', 'Protein IDs']
                      )
    return ds

def load_crohns_disease_dataset(path: Optional[Path] = None)->Dataset:
    """Crohn's disease dataset (PRIDE id: PXD000561). If path does not exist, download and extract a

    Args:
        path (Optional[Path]): path to either load dataset from or download dataset to.

    Returns:
        Dataset: The loaded dataset
    """
    if path is None:
        path = Path.home() / 'pyproteonet' / 'datasets' / 'PXD000561'
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