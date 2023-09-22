from typing import List, Optional, Callable, Union, Tuple, Dict
import warnings

import numpy as np
import pandas as pd

#from ..utils import load_external_data
from ..data.dataset_sample import DatasetSample
from ..data.dataset import Dataset
from ..data.molecule_set import MoleculeSet


def _de_duplicate(df: pd.DataFrame, ids: np.ndarray):
    df['id'] = ids
    return df.groupby('id').mean()

def read_dataset_pandas(dfs: Dict[str, pd.DataFrame], sample_columns: List[str], 
                        molecule_columns: Union[List[str], Dict[str, List[str]]] = [],
                        mappings: Union[List[Tuple[Tuple[str,str], Tuple[str,str]]], Dict[str, Tuple[Tuple[str,str], Tuple[str,str]]]] = [],
                        mapping_sep=',', value_name='abundance'
                       )->Dataset:
    if isinstance(molecule_columns, list):
        molecule_columns = {mol:molecule_columns for mol in dfs.keys()}
    if isinstance(mappings, list):
        mappings = {f'{mapping[0][0]}-{mapping[1][0]}':mapping for mapping in mappings}
    molecules = dict()
    maps = dict()
    for mol, df in dfs.items():
        molecules[mol] = df[molecule_columns[mol]]
    for map_name, ((mol1, col1),(mol2,col2)) in mappings.items():
        mapping = pd.DataFrame({mol1:dfs[mol1].index,
                                mol2:dfs[mol1][col1]})
        mapping[mol2] = mapping[mol2].str.split(mapping_sep)
        mapping = mapping.explode(mol2).reset_index(drop=True)
        id_mapper = pd.Series(index=dfs[mol2][col2], data=dfs[mol2][col2].index)
        mapping[mol2] = mapping[mol2].map(id_mapper)
        mapping.set_index([mol1,mol2], inplace=True, drop=True)
        maps[map_name] = mapping
    dataset = Dataset(molecule_set=MoleculeSet(molecules=molecules, mappings=maps))
    for sample in sample_columns:
        sample_values = {}
        for mol in dfs.keys():
            df = dfs[mol].loc[:, [sample]]
            df.rename(columns={sample:value_name}, inplace=True)
            sample_values[mol] = df
        dataset.create_sample(name=sample, values=sample_values)
    return dataset