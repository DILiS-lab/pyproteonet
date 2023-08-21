from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
import h5py

from ..data import MoleculeSet, Dataset

def load_alphapept_result(
    base_path: Union[str, Path],
    value_fields: List[str] = ['ms1_int_sum_apex'],
    molecules: Union[Dict[str, str], List[str]]={'protein_group':'protein_group', 'sequence':'peptide'},
    mappings: List[Tuple[str, str]] = [('protein_group', 'sequence')],
    skip_decoys: bool=True,
    keep_razor_mapping: bool=True,
)->Dataset:
    base_path = Path(base_path)
    database = h5py.File(base_path / "database.hdf", "r")
    protein_fdr = pd.read_hdf(base_path / "results.hdf", "protein_fdr")
    if skip_decoys:
        protein_fdr = protein_fdr[~protein_fdr.decoy]
    sequences = np.array(database["peptides"]["sequences"])
    prot_indices = np.array(database["peptides"]["protein_indices"])
    db_pointers = np.array(database["peptides"]["protein_indptr"])
    prots = pd.Series(database["proteins"]["name"]).iloc[prot_indices].astype(str).reset_index(drop=True)
    pep_pointers, prot_pointers = [], []
    for i in range(len(sequences)):
        target_prots = range(db_pointers[i], db_pointers[i + 1])
        prot_pointers.extend(target_prots)
        pep_pointers.extend([i] * len(target_prots))
    pep_prot_mapping = pd.DataFrame({"id": sequences[pep_pointers].astype(str), "map_id": prots.iloc[prot_pointers]})

    molecule_dfs = {}
    if isinstance(molecules, (list, tuple)):
        molecules = {mol:mol for mol in molecules}
    for mol, name in molecules.item():
        molecules[name] = pd.DataFrame(index = protein_fdr[mol].unique)
    for mapping in mappings:
        if 'protein_group' in mapping:
            pass
    #TODO: WIP

def load_alphapept_result(
    base_path: Union[str, Path],
    peptide_id_field: str="sequence",
    summed_peptide_fields: List[str]=["ms1_int_sum_apex"],
    averaged_peptide_fields: List[str]=[],
    per_sample_fields: List[str]=["ms1_int_sum_apex"],
    skip_decoys: bool=True,
    keep_razor_mapping: bool=True,
    keep_spectra: bool = True,
    samples: Optional[List[str]] = None,
)->Dataset:
    """Reads a Dataset from a directory containing results of the Alphapept analysis pipeline.

    Args:
        base_path (Union[str, Path]): Directory path to load files from
        peptide_id_field (str, optional): Field used as peptide identifier. Defaults to "precursor".
        summed_peptide_fields (List[str], optional): Fields whose values are summed when aggregating spectras to a peptide.
          Defaults to ["ms1_int_sum_apex"].
        averaged_peptide_fields (List[str], optional):  Fields whose values are averaged when aggregating spectras to a peptide.
          Defaults to ["mass"].
        per_sample_fields (List[str], optional): Fields that are not aggregated across samples (like abundance).
          All other fields in summed_peptide_fields and averaged_peptide_fields are per peptide fields and are only
          computed once for every peptide (aggregated across samples) Defaults to ["ms1_int_sum_apex"].
        skip_decoys (bool, optional): Skip peptides marked as decoy. Defaults to True.
        keep_razor_mapping (bool, optional): Generate a separate mapping which equals the mapping used by alphapepts quantification.
          In such a mapping every peptide is assigned to exactly one peptide group. If there are multiple possible peptide groups
          the peptide is assigned the group with the most found peptides (razor peptide). Defaults to True.
        keep_sptectra (bool, optional): Keep all spectra information as values under the key 'spectra'. Defaults to True.
        samples: (List[str], optional): Only load certain samples, if None load all samples. Defaults to None.

    Returns:
        Dataset: The loaded dataset
    """    
    base_path = Path(base_path)
    database = h5py.File(base_path / "database.hdf", "r")
    protein_fdr = pd.read_hdf(base_path / "results.hdf", "protein_fdr")
    if samples is not None:
        protein_fdr = protein_fdr[protein_fdr.sample_group.isin(samples)]
    if skip_decoys:
        protein_fdr = protein_fdr[~protein_fdr.decoy]
    sequences = np.array(database["peptides"]["sequences"])
    prot_indices = np.array(database["peptides"]["protein_indices"])
    db_pointers = np.array(database["peptides"]["protein_indptr"])
    prots = pd.Series(database["proteins"]["name"]).iloc[prot_indices].astype(str).reset_index(drop=True)
    pep_pointers, prot_pointers = [], []
    for i in range(len(sequences)):
        target_prots = range(db_pointers[i], db_pointers[i + 1])
        prot_pointers.extend(target_prots)
        pep_pointers.extend([i] * len(target_prots))
    pep_prot_mapping = pd.DataFrame({"id": sequences[pep_pointers].astype(str), "map_id": prots.iloc[prot_pointers]})

    protein_groups = pd.Series(protein_fdr.protein_group.unique())
    protein_groups.index = protein_groups
    protein_to_group_mapping = protein_groups.str.split(",").explode()
    protein_to_group_mapping = pd.Series(protein_to_group_mapping.index, index=protein_to_group_mapping)
    assert protein_to_group_mapping.index.is_unique

    per_sample_fields = set(per_sample_fields)
    groups = protein_fdr.groupby(peptide_id_field)
    peptides = groups[[f for f in summed_peptide_fields if f not in per_sample_fields]].sum()
    mean_fields = [f for f in averaged_peptide_fields if f not in per_sample_fields]
    peptides[mean_fields] = groups[mean_fields].mean()
    if keep_razor_mapping:
        assert (groups["protein_group"].nunique() == 1).all()
        peptides["protein_group"] = groups["protein_group"].first()

    #prot_mapping = pd.DataFrame({"id": protein_groups.to_numpy(), "map_id": protein_groups.to_numpy()})
    pep_mapping = pep_prot_mapping[pep_prot_mapping["map_id"].isin(protein_to_group_mapping.index)]
    if peptide_id_field != "sequence":
        assert (groups["sequence"].nunique() == 1).all()
        mapping_sequences = pd.DataFrame({"id": groups["sequence"].first()}).reset_index()
        pep_mapping = mapping_sequences.merge(pep_mapping, on="id", how="inner")
        del pep_mapping["id"]
        pep_mapping.rename(columns={peptide_id_field: "id"}, inplace=True)
    pep_mapping = pep_mapping[pep_mapping.id.isin(peptides.index)]
    pep_mapping.loc[:, "protein_group"] = protein_to_group_mapping.loc[pep_mapping.map_id].values
    del pep_mapping["map_id"]
    pep_mapping.rename(columns={'id':'peptide'}, inplace=True)
    pep_mapping.drop_duplicates(inplace=True)
    pep_mapping.set_index(['protein_group', 'peptide'], drop=True, inplace=True)

    sample_groups = protein_fdr.groupby(["sample_group", peptide_id_field])
    sample_peptides = sample_groups[[f for f in per_sample_fields if f in summed_peptide_fields]].sum()
    mean_fields = [f for f in per_sample_fields if f in averaged_peptide_fields]
    sample_peptides[mean_fields] = sample_groups[mean_fields].mean()

    molecules = {"protein_group": pd.DataFrame(index=protein_groups), "peptide": peptides}
    mappings = {"protein_group-peptide":  pep_mapping}
    if keep_razor_mapping:
        razor_mapping = peptides.reset_index().rename(columns={peptide_id_field: "peptide"})
        razor_mapping.set_index(['protein_group', 'peptide'], drop=True, inplace=True)
        #razor_mapping = {"protein_group": prot_mapping.copy(), "peptide": razor_mapping}
        mappings["razor"] = razor_mapping
    ms = MoleculeSet(molecules=molecules, mappings=mappings)
    ds = Dataset(molecule_set=ms)
    spec_groups = {}
    if keep_spectra:
        spec_groups = {sample:vals for sample, vals in protein_fdr.groupby('sample_group')}
    for name, values in sample_peptides.groupby("sample_group"):
        values.reset_index(level=["sample_group"], drop=True, inplace=True)
        values = {"peptide": values}
        if keep_spectra:
            values['spectrum'] = spec_groups[name]
        ds.create_sample(name=name, values=values)
    return ds
