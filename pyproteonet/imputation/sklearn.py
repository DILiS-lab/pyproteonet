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
    **kwargs
) -> Dataset:
    matrix = dataset.get_samples_value_matrix(molecule=molecule, value_column=column)
    matrix_imputed = imputation_function(matrix, **kwargs)
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
    imputer = KNNImputer(missing_values=dataset.missing_value, keep_empty_features=False, **kwargs)
    imputed = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
    )
    return imputed

def iterative_svd_impute(
    dataset: Dataset, molecule: str, column: str, min_value=0.001, **kwargs
) -> Dataset:
    imputer = IterativeImputer(missing_values=dataset.missing_value, keep_empty_features=False, min_value=min_value, **kwargs)
    imputed = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
    )
    return imputed
