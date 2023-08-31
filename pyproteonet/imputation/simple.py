from typing import Literal, Optional

import numpy as np
import pandas as pd

from ..data.dataset import Dataset

def across_sample_aggregate(dataset: Dataset, molecule: str, column: str, method: Literal['mean', 'median']='mean',
                            all_missing_percentile: Optional[float] = None, all_missing_constant:float=0):
    method = method.lower()
    if method=='mean':
        fn = np.nanmean
    elif method=='median':
        fn = np.nanmedian
    else:
        raise AttributeError(f'Method {method} not supported')
    matrix = dataset.get_samples_value_matrix(molecule=molecule, value_column=column)
    matrix_imputed = matrix.to_numpy()
    imp_value = fn(matrix_imputed, axis=1)
    missing_mask = np.isnan(matrix_imputed)
    imp_value = np.tile(imp_value[:,np.newaxis], (1,matrix_imputed.shape[1]))
    matrix_imputed[missing_mask] = imp_value[missing_mask]
    if all_missing_percentile is None:
        matrix_imputed[np.isnan(matrix_imputed)] = all_missing_constant
    else:
        percentile = np.nanpercentile(matrix_imputed, q=all_missing_percentile, axis=0)
        percentile = np.tile(percentile[np.newaxis,:], (matrix_imputed.shape[0],1))
        missing_mask = np.isnan(matrix_imputed)
        matrix_imputed[missing_mask] = percentile[missing_mask]
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns, index=matrix.index)
    assert matrix_imputed.shape == matrix.shape
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals

def min_det_impute(dataset, molecule: str, column: str, percentile=0.01):
    matrix = dataset.get_samples_value_matrix(molecule=molecule, value_column=column)
    matrix_imputed = matrix.to_numpy()
    percentile = np.nanpercentile(matrix_imputed, q=percentile, axis=0)
    percentile = np.tile(percentile[np.newaxis,:], (matrix_imputed.shape[0],1))
    missing_mask = np.isnan(matrix_imputed)
    matrix_imputed[missing_mask] = percentile[missing_mask]
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns, index=matrix.index)
    assert matrix_imputed.shape == matrix.shape
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals