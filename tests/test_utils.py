import os
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

from pyproteonet.data import Dataset, MoleculeSet
from pyproteonet.io import datasets

TESTDATA_DIR = Path(os.path.dirname(__file__)) / 'testdata'


def load_maxlfq_benchmark(path: Path = TESTDATA_DIR / 'maxlfq_benchmark')->Dataset:
    return datasets.load_maxlfq_benchmark(path=path)


def create_toy_dataset()->Dataset:
    proteins = pd.DataFrame(index=['A', 'B', 'C', 'D'])
    peptides = pd.DataFrame(index=list(range(13)))
    peptide_values_sample1 = {
        0:0.5,
        1:3.0,
        2:2.0,
        3:1.0,
        4:4.5,
        5:2.0,
        6:2.0,
        7:5.5,
        8:2.0,
        9:5.0,
        10:5.0,
        11:2.0,
        12:1.0,
    }
    peptide_values_sample1 = pd.DataFrame({'abundance':pd.Series(peptide_values_sample1)})
    peptide_values_sample2 = peptide_values_sample1 * 2
    peptide_protein_mapping = [(0, 'A'), (1, 'A'), (2, 'A'), (3, 'A'),  (4, 'A'),  (7, 'A'),
                               (4, 'B'), (7, 'B'), (5, 'B'), (6, 'B'), 
                               (8, 'C'), (9, 'C'), (10, 'C'), (11, 'C'), (12, 'C')
                              ]
    peptide_protein_mapping = pd.DataFrame(peptide_protein_mapping, columns=['peptide', 'protein'])
    peptide_protein_mapping.set_index(['peptide', 'protein'], inplace=True)
    ms = MoleculeSet(molecules={'protein':proteins, 'peptide':peptides},
                     mappings = {'peptide-protein': peptide_protein_mapping})
    ds = Dataset(molecule_set=ms)
    ds.create_sample('sample1', values={'peptide': peptide_values_sample1})
    ds.create_sample('sample2', values={'peptide': peptide_values_sample2})
    return ds
    
        