from typing import List, Union, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from pyproteonet.data import MoleculeSet, Dataset


def load_maxquant(
    peptides_table: Union[pd.DataFrame, str, Path],
    samples: Optional[List[str]] = None,
    protein_groups_table: Optional[Union[pd.DataFrame, str, Path]] = None,
    protein_group_value_columns: List[str] = ["Intensity", "iBAQ", "LFQ intensity"],
    peptide_value_columns: List[str] = ["Intensity"],
    peptide_columns: List[str] = ["Sequence"],
    protein_group_columns: List[str] = ["Fasta headers"],
    missing_value: float = 0
) -> Dataset:
    """Loads a dataset given in MaxQuant format (experimental for now). Might not support all datsets.

    Args:
        protein_groups_table (Union[pd.DataFrame, str, Path]): Pandas dataframe or path to proteinGroups.txt.
        peptides_table (Union[pd.DataFrame, str, Path]): Pandas dataframe or path to peptides.txt.
        samples (List[str], optional): List of sample names to load (must be present as columns in peptides.txt and proteinGroups.txt).
        protein_group_value_columns (List[str], optional): Values to load for every protein group and sample. 
            Sample name and value column will be concatenated to a column name which is then looked up n the peptides table.
            Defaults to ["Intensity", "iBAQ", "LFQ intensity"].
        peptide_value_columns (List[str], optional): Values to load for every protein group and sample. 
            Sample name and value column will be concatenated to a column name which is then looked up in the protein groups table.
            Defaults to ["Intensity"].
        peptide_columns (List[str], optional): Columns from the peptides table to keep as meta info per molecule. Defaults to ["Sequence"].
        protein_group_columns (List[str], optional): Columns from the protein group table to keep as meta info per molecule.
            Defaults to ["Fasta headers"].
        missing_vlue (float, optional): Value interpreted as missing, Defaults to 0.

    Returns:
        Dataset: The loaded Dataset.
    """    
    if isinstance(peptides_table, (str, Path)):
        peptides_table = pd.read_csv(peptides_table, sep="\t")
    if samples is None:
        samples = []
        for value_column in ['Intensity']:
            value_samples = peptides_table.columns.str.extract(f'{value_column} (.+)')
            samples.append(set(value_samples.loc[~value_samples.loc[:,0].isna(), 0]))
        samples = list(set.intersection(*samples))
        samples.sort()
    if protein_groups_table is None:
        protein_groups = peptides_table.loc[:, 'Protein group IDs'].astype(str).str.split(';').explode().unique().astype(int)
        protein_groups = pd.DataFrame(index=pd.Index(protein_groups, name='id')) 
    elif isinstance(protein_groups_table, (str, Path, pd.DataFrame)):
        if isinstance(protein_groups_table, (str, Path)):
            protein_groups_table = pd.read_csv(protein_groups_table, sep="\t")
        protein_groups = protein_groups_table.loc[:, protein_group_columns]
    peptides = peptides_table.loc[:, peptide_columns].copy()
    map = peptides_table["Protein group IDs"].astype(str).str.split(";").explode().astype(int)
    map = map.reset_index().rename(columns={"index": "peptide", "Protein group IDs": "protein_group"})
    #import pdb; pdb.set_trace()
    map.set_index(['peptide', 'protein_group'], inplace=True, drop=True)
    # mapping_protein_group = {
    #     "protein_group": pd.DataFrame({"id": protein_groups.index, "map_id": protein_groups.index}),
    #     "peptide": map,
    # }
    ms = MoleculeSet(
        molecules={"peptide": peptides, "protein_group": protein_groups},
        mappings={"peptide-protein_group": map},
    )
    ds = Dataset(molecule_set=ms)
    for sample in samples:
        values = dict()
        columns = {f"{v} {sample}":v for v in peptide_value_columns}
        peptide_values = peptides_table.loc[:, columns.keys()].rename(columns=columns)
        peptide_values[peptide_values==missing_value] = np.nan
        values['peptide'] = peptide_values
        if protein_groups_table is not None:
            columns = {f"{v} {sample}":v for v in protein_group_value_columns}
            protein_group_values = protein_groups_table.loc[:, columns.keys()].rename(columns=columns)
            protein_group_values[protein_group_values==missing_value] = np.nan
            values['protein_group'] = protein_group_values
        ds.create_sample(name=sample, values=values)
    return ds
