from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import pandas as pd

# from ..utils import load_external_data
from ..data.dataset import Dataset
from ..data.molecule_set import MoleculeSet


def _de_duplicate(df: pd.DataFrame, ids: np.ndarray):
    df["id"] = ids
    return df.groupby("id").mean()


def read_mapped_dataframe(
    df: pd.DataFrame,
    molecule: str,
    sample_columns: List[str],
    id_column: Optional[str] = None,
    result_column_name: str = "abundance",
    mapping_column: Optional[str] = None,
    mapping_sep: str = ",",
    mapping_molecule: str = "protein",
    mapping_name="peptide-protein",
) -> Dataset:
    """
    Reads a dataframe containing a mapping as column with comma separated ids of the mapped molecule.

    Args:
        df (pd.DataFrame): The input dataframe.
        molecule (str): The name of the molecule column in the dataframe.
        sample_columns (List[str]): The list of sample columns in the dataframe.
        id_column (Optional[str], optional): The name of the ID column in the dataframe. Defaults to None.
        result_column_name (str, optional): The name of the result column in the dataframe. Defaults to "abundance".
        mapping_column (Optional[str], optional): The name of the mapping column in the dataframe. Defaults to None.
        mapping_sep (str, optional): The separator used in the mapping column. Defaults to ",".
        mapping_molecule (str, optional): The name of the molecule in the mapping column. Defaults to "protein".
        mapping_name (str, optional): The name of the mapping. Defaults to "peptide-protein".

    Returns:
        Dataset: The converted Dataset object.
    """
    if id_column is None:
        mols = df.loc[:, []].copy()
        mols.index.name = 'id'
        molecules = {molecule: mols}
    else:
        index = pd.Index(df.loc[:, id_column])
        index.name = 'id'
        mols = pd.DataFrame(index=index)
        molecules = {molecule: mols}
    mappings = {}
    if mapping_column is not None:
        mapping_mols = df[mapping_column].str.split(mapping_sep).explode()
        mapping = pd.DataFrame({molecule: df.loc[mapping_mols.index].index if id_column is None else df.loc[mapping_mols.index][id_column], mapping_molecule: mapping_mols})
        mappings[mapping_name] = mapping.set_index([molecule, mapping_molecule])
        molecules[mapping_molecule] = pd.DataFrame(index=mapping_mols, columns=[])
    ms = MoleculeSet(molecules=molecules, mappings=mappings)
    dataset = Dataset(molecule_set=ms, )
    for c in sample_columns:
        vals = pd.DataFrame(index=mols.index, data={result_column_name:df.loc[:, c].values})
        dataset.create_sample(name=c, values={molecule: vals})
    return dataset


def read_multiple_mapped_dataframes(
    dfs: Dict[str, pd.DataFrame],
    sample_columns: List[str],
    molecule_columns: Union[List[str], Dict[str, List[str]]] = [],
    mappings: Union[
        List[Tuple[Tuple[str, str], Tuple[str, str]]],
        Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]],
    ] = [],
    mapping_sep=",",
    value_name="abundance",
) -> Dataset:
    """
    Reads multiple mapped dataframes each containing a mapping column with lists of ids of mapped molecules.

    Args:
        dfs (Dict[str, pd.DataFrame]): A dictionary of dataframes, where the keys represent the molecule names and the values represent the dataframes.
        sample_columns (List[str]): A list of column names representing the samples.
        molecule_columns (Union[List[str], Dict[str, List[str]]], optional): The column names representing the non-sample-specific molecule columns (e.g. sequence). It can be a list if the same columns are used for all molecules, or a dictionary if different columns are used for different molecules. Defaults to an empty list.
        mappings (Union[List[Tuple[Tuple[str, str], Tuple[str, str]]], Dict[str, Tuple[Tuple[str, str], Tuple[str, str]]]], optional): The mappings between molecules. It can be a list of tuples or a dictionary from mapping name to tuple. Each tuple represents a mapping given by two tuples, each of which contains the key of the dataframe and the name of the mapping column (ids from both mapping columns are matched to find the mapped molecules) . Defaults to an empty list.
        mapping_sep (str, optional): The separator used in the mapping columns to separed ids. Defaults to ",".
        value_name (str, optional): The name of the create value column. Defaults to "abundance".

    Returns:
        Dataset: A Dataset object.
    """
    if isinstance(molecule_columns, list):
        molecule_columns = {mol: molecule_columns for mol in dfs.keys()}
    if isinstance(mappings, list):
        mappings = {f"{mapping[0][0]}-{mapping[1][0]}": mapping for mapping in mappings}
    molecules = dict()
    maps = dict()
    for mol, df in dfs.items():
        molecules[mol] = df[molecule_columns[mol]]
    for map_name, ((mol1, col1), (mol2, col2)) in mappings.items():
        mapping = pd.DataFrame({mol1: dfs[mol1].index, mol2: dfs[mol1][col1]})
        mapping[mol2] = mapping[mol2].str.split(mapping_sep)
        mapping = mapping.explode(mol2).reset_index(drop=True)
        id_mapper = pd.Series(index=dfs[mol2][col2], data=dfs[mol2][col2].index)
        mapping[mol2] = mapping[mol2].map(id_mapper)
        mapping.set_index([mol1, mol2], inplace=True, drop=True)
        maps[map_name] = mapping
    dataset = Dataset(molecule_set=MoleculeSet(molecules=molecules, mappings=maps))
    for sample in sample_columns:
        sample_values = {}
        for mol in dfs.keys():
            df = dfs[mol].loc[:, [sample]]
            df.rename(columns={sample: value_name}, inplace=True)
            sample_values[mol] = df
        dataset.create_sample(name=sample, values=sample_values)
    return dataset
