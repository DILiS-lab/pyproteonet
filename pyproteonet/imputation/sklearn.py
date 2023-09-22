from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from ..data.dataset import Dataset

def generic_matrix_imputation(
    dataset: Dataset,
    molecule: str,
    column: str,
    imputation_function: Callable[[np.ndarray], np.ndarray],
    transpose: bool = False,
    **kwargs
) -> pd.Series:
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    if transpose:
        matrix = matrix.to_numpy().T
    matrix_imputed = imputation_function(matrix, **kwargs)
    if transpose:
        matrix_imputed = matrix_imputed.T
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns, index=matrix.index)
    assert matrix.shape == matrix_imputed.shape
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals
    #dataset.set_samples_value_matrix(matrix=matrix_imputed, molecule=molecule, column=result_column)
    #return dataset

def knn_impute(
    dataset: Dataset, molecule: str, column: str, **kwargs
) -> Dataset:
    imputer = KNNImputer(missing_values=dataset.missing_value, **kwargs)
    imputed = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
    )
    return imputed

def iterative_impute(
    dataset: Dataset, molecule: str, column: str, min_value:Optional[float]=None, max_value:Optional[float]=None, **kwargs
) -> pd.Series:
    if min_value is None:
        min_value = dataset.values[molecule][column].min()
    if max_value is None:
        max_value = dataset.values[molecule][column].max()
    imputer = IterativeImputer(missing_values=dataset.missing_value, keep_empty_features=False,
                               min_value=min_value, max_value=max_value, **kwargs)
    imputed = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
    )
    return imputed
