from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from ..data.dataset import Dataset

def generic_matrix_imputation(
    dataset: Dataset,
    molecule: str,
    column: str,
    imputation_function: Callable[[np.ndarray], np.ndarray],
    result_column: Optional[str] = None,
    inplace: bool = False,
    **kwargs
) -> Dataset:
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    matrix_imputed = imputation_function(matrix, **kwargs)
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns)
    dataset.set_samples_value_matrix(matrix=matrix_imputed, molecule=molecule, column=result_column)
    return dataset

def knn_imputation(
    dataset: Dataset, molecule: str, column: str, result_column: Optional[str] = None, inplace: bool = False, **kwargs
) -> Dataset:
    imputer = KNNImputer(missing_values=dataset.missing_value, **kwargs)
    dataset = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
        result_column=result_column,
        inplace=inplace,
    )
    return dataset
