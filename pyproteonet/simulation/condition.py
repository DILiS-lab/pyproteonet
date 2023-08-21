from typing import Optional, List

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .utils import get_numpy_random_generator
from ..data.dataset import Dataset


def add_simulated_condition(dataset: Dataset, condition_affected_samples: List[str], random_state = None,
                            input_column: str = 'abundance', output_column: str = 'condition_abundance', mapping: str = 'gene',
                            condition_affected_frac: Optional[float] = 0.15, condition_affected_proteins: Optional[List] = None,
                            condition_log2_mean: float = 0, condition_log2_std: float = 1,
                            error_log2_mean: float = 0, error_log2_std: float = 0.1, inplace: bool = False) -> Dataset:
    if condition_affected_frac is None and condition_affected_proteins is None:
        raise AttributeError("Either frac_condition_affected or condition_proteins must be set!")
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
        if (peptide_values[input_column]<=0).any() or (protein_values[input_column]<=0).any():
            raise ValueError('All non missing input values should must be positive!')
        if sample_name in condition_affected:
            #save the original protein values in mapped so that we do not overwrite them if input_column==output_column
            mapped['prot_abundance'] = protein_values.loc[mapped.protein, input_column].values
            error = rng.normal(loc=error_log2_mean, scale=error_log2_std, size=len(condition_proteins))
            protein_values[output_column] = protein_values[input_column]
            protein_values.loc[condition_proteins.index, output_column] *= 2**(condition_proteins['condition_effect'] + error)
            protein_values[output_column][protein_values[output_column]<=0] = np.nan
        
            mapped['pep_abundance'] = peptide_values.loc[mapped.peptide, input_column].values
            pep_sum = mapped.groupby('peptide').prot_abundance.sum(numeric_only=True)
            mapped['pep_prot_sum'] = pep_sum.loc[mapped.peptide].values
            mapped['pep_scaling_factor'] = mapped['prot_abundance'] / mapped['pep_prot_sum']
            pep_error = rng.normal(loc=error_log2_mean, scale=error_log2_std, size=sample.molecules['peptide'].shape[0])
            pep_error = pd.Series(pep_error, index=sample.molecules['peptide'].index)
            mapped['pep_error'] = pep_error.loc[mapped['peptide']].values
            cond_mapped = mapped.loc[mapped_condition_mask]
            mapped['mask'] = mapped_condition_mask
            mapped.loc[:, 'cond_scaling_factor'] = 1
            mapped.loc[mapped_condition_mask, 'cond_scaling_factor'] = 2**(condition_proteins.loc[cond_mapped['protein'], 'condition_effect'].values + cond_mapped['pep_error'])
            mapped['res'] = mapped['pep_scaling_factor'] * mapped['pep_abundance'] * mapped['cond_scaling_factor']
            peptide_values[output_column] = mapped.groupby('peptide').res.sum()
            peptide_values[output_column][peptide_values[output_column]<=0] = np.nan
        else:
            protein_values[output_column] = protein_values[input_column]
            peptide_values[output_column] = peptide_values[input_column]
    return dataset