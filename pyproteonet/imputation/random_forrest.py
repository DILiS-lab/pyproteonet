from typing import Optional, Callable
import sys

import numpy as np
import pandas as pd
import sklearn

from .missingpy import MissForest
from ..data.dataset import Dataset
from .sklearn import generic_matrix_imputation

# def missing_forrest_impute(
#     dataset: Dataset, molecule: str, column: str, transpose:bool = False, **kwargs
# ) -> Dataset:
#     imputer = MissForest(missing_values=dataset.missing_value, **kwargs)
#     dataset = generic_matrix_imputation(
#         dataset=dataset,
#         molecule=molecule,
#         column=column,
#         imputation_function=imputer.fit_transform,
#         transpose = transpose,
#     )
#     return dataset

def missing_forrest_impute(
    dataset: Dataset, molecule: str, column: str, molecules_as_variables:bool = False, result_column: Optional[str] = None, **kwargs
) -> Dataset:
    imputer = MissForest(missing_values=dataset.missing_value, **kwargs)
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    mask = ~matrix.isna().all(axis=1)
    #mask = np.full(matrix.shape[0], True)#

    mat = matrix.loc[mask, :].to_numpy()
    if molecules_as_variables:
        mat = mat.T
    mat = imputer.fit_transform(mat)
    if molecules_as_variables:
        mat = mat.T
    assert mat.shape[0] == mask.sum()
    assert mat.shape[1] == matrix.shape[1]

    matrix.loc[mask, :] = mat
    col_means = matrix.mean()
    for c in matrix:
        col = matrix[c]
        col[col.isna()] = col_means[c]

    if result_column is not None:
        dataset.set_samples_value_matrix(molecule=molecule, column=result_column, matrix=matrix)
    
    vals = matrix.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals