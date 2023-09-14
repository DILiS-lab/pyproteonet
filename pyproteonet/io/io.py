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

def read_protein_peptide_dataset_pandas(protein_df: pd.DataFrame, peptide_df: pd.DataFrame, sample_columns: List[str],
                                        mapping_columns = ['gene'], protein_id_column = 'ProteinID', peptide_id_column = 'Sequence',
                                        protein_mapping_transform: Optional[Callable] = None,
                                        peptide_mapping_transform: Optional[Callable] = None):
    molecules = {}
    values = {}
    for sample_name in sample_columns:
        values[sample_name] = {}
    molecules['protein'] = pd.DataFrame(index=pd.Index(protein_df[protein_id_column].unique(), name='id'))
    protein_values = protein_df.loc[:, [protein_id_column] + sample_columns].rename(columns={protein_id_column:'id'}).set_index('id')
    for sample_name in sample_columns:
        vals = protein_values[[sample_name]].rename(columns={sample_name:'abundance'})
        values[sample_name]['protein'] = vals
    molecules['peptide'] = pd.DataFrame(index=pd.Index(peptide_df[peptide_id_column].unique(), name='id'))
    peptide_values = peptide_df.loc[:, [peptide_id_column] + sample_columns].rename(columns={peptide_id_column:'id'}).set_index('id')
    for sample_name in sample_columns:
            vals = peptide_values[[sample_name]].rename(columns={sample_name:'abundance'})
            values[sample_name]['peptide'] = vals
    mappings = {}
    for mapping_column in mapping_columns:
        protein_mapping = pd.DataFrame({'id':protein_df[protein_id_column],
                                         'map_id':protein_df[mapping_column]})
        protein_mapping['map_id'] = protein_mapping['map_id'].str.split(',')
        protein_mapping = protein_mapping.explode('map_id').reset_index(drop=True)
        if protein_mapping_transform is not None:
            mapped = protein_mapping_transform(protein_mapping.map_id)
            protein_mapping = pd.DataFrame(data={'map_id':mapped, 'id':protein_mapping.loc[mapped.index, 'id']})
            protein_mapping = protein_mapping.reset_index(drop=True)
        peptide_mapping = pd.DataFrame({'id':peptide_df[peptide_id_column],
                                         'map_id':peptide_df[mapping_column]})
        peptide_mapping['map_id'] = peptide_mapping['map_id'].str.split(',')
        peptide_mapping = peptide_mapping.explode('map_id').reset_index(drop=True)
        if peptide_mapping_transform is not None:
            mapped = peptide_mapping_transform(peptide_mapping.map_id)
            peptide_mapping = pd.DataFrame(data={'map_id':mapped, 'id':peptide_mapping.loc[mapped.index, 'id']})
            peptide_mapping = peptide_mapping.reset_index(drop=True)
        protein_mapping.rename(columns={"id": 'protein'}, inplace=True)
        peptide_mapping.rename(columns={"id": 'peptide'}, inplace=True)
        mapping =  protein_mapping.merge(peptide_mapping, on="map_id", how="inner").drop(columns=['map_id'])
        if mapping.duplicated().any():
            warnings.warn(f"The mapping {mapping_column} resulted in duplicated edges which will be removed.")
            mapping = mapping.drop_duplicates()
        mapping.set_index(['protein', 'peptide'], drop=True, inplace=True)
        mapping = mapping.loc[mapping.index.get_level_values('protein').isin(molecules['protein'].index) &
                              mapping.index.get_level_values('peptide').isin(molecules['peptide'].index), :]
        mappings[mapping_column] = mapping
    molecule_set = MoleculeSet(molecules=molecules, mappings=mappings)
    dataset = Dataset(molecule_set=molecule_set)
    for sample_name in sample_columns:
        dataset.create_sample(name=sample_name, values=values[sample_name])
    return dataset

def read_dataset_tsv(base_path: str, sample_columns: List[str], 
                     peptide_suffix = '_peptides.tsv', protein_suffix = '_prot.tsv',
                     mapping_column = 'gene', protein_id_column = 'ProteinID', peptide_id_column = 'Sequence'):
    protein_df = pd.read_csv(base_path + protein_suffix, sep='\t')
    peptide_df = pd.read_csv(base_path + peptide_suffix, sep='\t')
    return  read_protein_peptide_dataset_pandas(protein_df=protein_df, peptide_df=peptide_df, sample_columns=sample_columns,
                                 mapping_column=mapping_column, protein_id_column=protein_id_column,
                                 peptide_id_column=peptide_id_column)