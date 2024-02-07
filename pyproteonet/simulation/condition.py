from typing import Iterable, Optional, List, Union
import collections

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .utils import get_numpy_random_generator
from ..data.dataset import Dataset


def introduce_random_condition(
    dataset: Dataset,
    molecule: str,
    column: str,
    affected: Union[float, int, Iterable] = 0.15,
    log2_cond_factor_mean: float = 0,
    log2_cond_factor_std: float = 1,
    samples: Union[List[str], int, float] = 0.5,
    result_column: Optional[str] = None,
    inplace: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None
) -> Dataset:
    """Samples condition factors for a subset of randomly chosen molecues
       and multiplies the molecules values of affected with those factors.

    Args:
        dataset (Dataset): Input Dataset.
        amount_affected (Union[float, int, Iterable], optional): Number of affected molecules or indices of affected molecules.
            Floats in [0, 1.0] are intepreted as relative fractions, integers are intepreted as absolute values.
            If an iterable is given it will be interpreted as the molecule indices of the condition affected molecules. Defaults to 0.15.
        log2_cond_factor_mean (float, optional): Mean of normal distribution in log2 space,
            used for sampling conditin factors. Defaults to 0.
        log2_cond_factor_std (float, optional): Std of normal distribution in log2 space,
            used for sampling conditin factors. Defaults to 1.
        samples (Union[List[str], int, float], optional): If list, will be interpreted as names of samples,
            if float will be interpreted as fraction of samples, if int will be interpreted as number of fractions.
            Defaults to 0.5.
        molecule (str, optional): Molecule type to draw values for. Defaults to 'protein'.
        column (str, optional): Column to apply condition to Defaults to 'abundance'.
        result_column (str, optional): Column to write result to. Defaults to the input column if not given.
        inplace (float, optional): Whether to copy the dataset before introducing the condition.
        random_seed (_type_, optional): Seed for random generator. Defaults to None.


    Returns:
        Dataset:  The resulting Dataset with condition applied.
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)
    condition_samples = dataset.sample_names
    if isinstance(samples, list):
        condition_samples = samples
    elif isinstance(samples, (int, float)):
        if isinstance(samples, float) and samples <= 1.0:
            samples = int(len(condition_samples) * samples)
        condition_samples = rng.choice(condition_samples, size=int(samples))
    condition_samples = set(condition_samples)
    ds_sample_names = set(dataset.sample_names)
    for sample_name in condition_samples:
        if sample_name not in ds_sample_names:
            raise ValueError(f"Condition sample not found! The sample  '{sample_name}' is not in the dataset.")
    if isinstance(affected, collections.abc.Iterable):
        condition_affected_mols = affected
    else:
        if isinstance(affected, float):
            affected = int(len(dataset.molecules[molecule]) * affected)
        condition_affected_mols = rng.choice(
            dataset.molecules[molecule].index,
            size=affected,
            replace=False,
        )
    factor = rng.normal(
            loc=log2_cond_factor_mean,
            scale=log2_cond_factor_std,
            size=len(condition_affected_mols),
        )
    factor = 2**factor
    for sample_name in dataset.sample_names:
        sample = dataset[sample_name]
        vals = sample.values[molecule][column].copy()
        if sample_name in condition_samples:
            vals.loc[condition_affected_mols] *= factor
        sample.values[molecule][result_column] = vals
    return dataset

def add_simulated_condition(dataset: Dataset, condition_affected_samples: List[str],
                            column: str, mapping: str, result_column: Optional[str] = None,
                            condition_affected_frac: Optional[float] = 0.15, condition_affected_proteins: Optional[List] = None,
                            condition_log2_mean: float = 0, condition_log2_std: float = 1,
                            error_log2_mean: float = 0, error_log2_std: float = 0.1, inplace: bool = False,
                            condition_factor_base: float = 2, random_state = None) -> Dataset:
    """
    Adds a simulated condition to an already existing dataset (a mapping is considered and the condition is introduced to both mapping partners).

    Args:
        dataset (Dataset): The dataset to modify.
        condition_affected_samples (List[str]): List of sample names that should be affected by the condition.
        column (str): The column name containing the original values.
        mapping (str): The mapping between protein and peptide.
        result_column (Optional[str]): The column name to store the modified values. If None, it will be the same as the original column.
        condition_affected_frac (Optional[float]): The fraction of condition-affected proteins. Default is 0.15.
        condition_affected_proteins (Optional[List]): List of specific condition-affected proteins. If None, proteins will be randomly selected based on the fraction. Default is None.
        condition_log2_mean (float): The mean of the condition effect in log2 scale. Default is 0.
        condition_log2_std (float): The standard deviation of the condition effect in log2 scale. Default is 1.
        error_log2_mean (float): The mean of the error in log2 scale. Default is 0.
        error_log2_std (float): The standard deviation of the error in log2 scale. Default is 0.1.
        inplace (bool): Whether to modify the dataset in-place or create a copy. Default is False.
        condition_factor_base (float): The base of the condition factor. Default is 2.
        random_state: Random seed for reproducibility. Default is None.

    Returns:
        Dataset: The modified dataset.
    """
    if condition_affected_frac is None and condition_affected_proteins is None:
        raise AttributeError("Either frac_condition_affected or condition_proteins must be set!")
    if result_column is None:
        result_column = column
    if not inplace:
        dataset = dataset.copy()
    condition_affected = set(condition_affected_samples)
    rng = get_numpy_random_generator(seed=random_state)
    proteins = dataset.molecules['protein'].copy()
    if condition_affected_proteins is not None:
        condition_proteins = proteins.loc[condition_affected_proteins]
    else:
        condition_proteins = proteins.sample(frac=condition_affected_frac, random_state=random_state)
    dataset.molecules['protein'].loc[:, 'condition_affected'] = False
    dataset.molecules['protein'].loc[condition_proteins.index, 'condition_affected'] = True

    pep_map = dataset.molecule_set.get_mapped_pairs(molecule_a='protein', molecule_b='peptide', mapping=mapping)
    pep_map['prot_affected'] = dataset.molecules['protein'].loc[pep_map.protein, 'condition_affected'].values
    peps_affected = pep_map.groupby('peptide').prot_affected.sum()
    peps_affected = peps_affected[peps_affected > 0]
    dataset.molecules['peptide'].loc[:, 'condition_affected'] = False
    dataset.molecules['peptide'].loc[peps_affected.index, 'condition_affected'] = True

    condition_effect = np.zeros(len(condition_proteins))
    mask = np.ones(len(condition_proteins), dtype=bool)
    while mask.sum() > 0:
        condition_effect[mask] = rng.normal(loc=condition_log2_mean, scale=condition_log2_std, size=mask.sum())
        mask = np.abs(condition_effect) < 1
    condition_proteins['condition_effect'] = condition_effect
    #proteins.loc[condition_proteins.index, 'condition_effect'] = condition_effect
    mapped = dataset.molecule_set.get_mapped_pairs(molecule_a="protein", molecule_b="peptide", mapping=mapping)
    mapped_condition_mask = mapped.protein.isin(condition_proteins.index)
    for sample_name, sample in tqdm(dataset.samples_dict.items()):
        protein_values = sample.values['protein']
        peptide_values = sample.values['peptide']
        if (peptide_values[column]<=0).any() or (protein_values[column]<=0).any():
            raise ValueError('All non-missing input values should must be positive!')
        if sample_name in condition_affected:
            #save the original protein values in mapped so that we do not overwrite them if input_column==output_column
            mapped['prot_abundance'] = protein_values.loc[mapped.protein, column].values
            error = rng.normal(loc=error_log2_mean, scale=error_log2_std, size=len(condition_proteins))
            protein_values[result_column] = protein_values[column]
            protein_values.loc[condition_proteins.index, result_column] *= condition_factor_base**(condition_proteins['condition_effect'] + error)
            protein_values[result_column][protein_values[result_column]<=0] = np.nan
        
            mapped['pep_abundance'] = peptide_values.loc[mapped.peptide, column].values
            pep_sum = mapped.groupby('peptide').prot_abundance.sum(numeric_only=True)
            mapped['pep_prot_sum'] = pep_sum.loc[mapped.peptide].values
            mapped['pep_scaling_factor'] = mapped['prot_abundance'] / mapped['pep_prot_sum']
            pep_error = rng.normal(loc=error_log2_mean, scale=error_log2_std, size=sample.molecules['peptide'].shape[0])
            pep_error = pd.Series(pep_error, index=sample.molecules['peptide'].index)
            mapped['pep_error'] = pep_error.loc[mapped['peptide']].values
            cond_mapped = mapped.loc[mapped_condition_mask]
            mapped['mask'] = mapped_condition_mask
            mapped.loc[:, 'cond_scaling_factor'] = 1
            mapped.loc[mapped_condition_mask, 'cond_scaling_factor'] = condition_factor_base**(condition_proteins.loc[cond_mapped['protein'], 'condition_effect'].values + cond_mapped['pep_error'])
            mapped['res'] = mapped['pep_scaling_factor'] * mapped['pep_abundance'] * mapped['cond_scaling_factor']
            peptide_values[result_column] = mapped.groupby('peptide').res.sum()
            peptide_values[result_column][peptide_values[result_column]<=0] = np.nan
        else:
            protein_values[result_column] = protein_values[column]
            peptide_values[result_column] = peptide_values[column]
    return dataset