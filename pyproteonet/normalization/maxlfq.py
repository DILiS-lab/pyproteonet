from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..data import Dataset

def get_total_error(normalization: np.ndarray, profiles: np.ndarray) -> float:
    """Computes the summed peptide errors over the whole dataset.

    Args:
        normalization (np.ndarray): per sample normalization factors.
        profiles (np.ndarray): peptide intensity profiles over the dataset.

    Returns:
        float: summed peptide error.
    """
    normalization = normalization.reshape(profiles.shape[:2])
    reshaped_profiles = np.transpose(profiles, (2, 0, 1)) # move axes to support broadcasting
    normalized_profiles = reshaped_profiles*normalization
    pep_ints_all = np.nansum(normalized_profiles, axis=1) 
    #returns upper triangular indices, with offset of 1, which gives us every pair of indices in our array
    i, j = np.triu_indices(pep_ints_all.shape[1], k=1) #returns upper triangular indices
    x = pep_ints_all[:,i]
    y = pep_ints_all[:,j]
    #this operation will result in pos/neg infinity for zero values in x or y, and will cause expected runtime warnings
    with np.errstate(divide = 'ignore', invalid='ignore'):
        errors = np.log(x/y)
    #change pos/neg infinity values to 0 to compute accurate sum
    errors = np.nan_to_num(errors, posinf=0, neginf=0)
    #absolute value is not necessary as squaring should always result in a non negative value
    total_error = (errors**2).sum()
    #total_error = np.abs(errors).sum()
    return total_error

def normalize_experiment_SLSQP(profiles: np.ndarray) -> np.ndarray:
    """Calculates normalization with SLSQP approach.
    Args:
        profiles (np.ndarray): peptide intensities.

    Returns:
        np.ndarray: normalization factors.
    """
    x = profiles / np.sum(profiles, axis=1, keepdims=1) #all signal from one fraction, protein and run
    median_estimate = 1/np.nanmedian(x, axis=2, keepdims=1) #median of all proteins, inverse
    x0 = median_estimate/np.max(median_estimate, axis=1, keepdims=1) #normalize
    x0 = x0.reshape(profiles.shape[0] * profiles.shape[1])
    x0 = np.nan_to_num(x0, nan=1)
    bounds = [(0.01, 1) for _ in x0]
    res = minimize(get_total_error, args = profiles , x0 = x0, bounds=bounds, method='SLSQP', options={'disp': False, 'maxiter':100*profiles.shape[1]})
    solution = res.x/np.max(res.x)
    solution = solution.reshape(profiles.shape[:2])
    return solution

def normalize_maxlfq(dataset: Dataset, molecule: str, column: str, reference_ids: Optional[pd.Index]=None, result_column: Optional[str] = None)->pd.DataFrame:
    pep_table = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    if reference_ids is not None:
        tab = pep_table.loc[reference_ids].to_numpy()
    else:
        tab = pep_table.to_numpy()
    tab = tab.T[np.newaxis, :, :]
    factors = normalize_experiment_SLSQP(tab)
    print(factors)
    for column, fac in zip(pep_table.columns, factors.flatten()):
        pep_table[column] *= fac
    res = pep_table.stack().swaplevel()
    res.index.set_names(['sample', 'id'], inplace=True)
    if result_column is not None:
        dataset.values[molecule][result_column] = res
    return res
