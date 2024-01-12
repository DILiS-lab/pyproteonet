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
    result_column: Optional[str] = None,
    **kwargs
) -> pd.Series:
    """Apply a scikit-learn matrix imputer to a dataset.

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
        column (str): Name of the value column to impute.
        imputation_function (Callable[[np.ndarray], np.ndarray]): Impute function working on a 2d numpy array.
        transpose (bool, optional): Whether to transpose the matrix before imputation. Defaults to False.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.

    Returns:
        pd.Series: The imputed values.
    """
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
    if result_column is not None:
        dataset.values[molecule][result_column] = vals
    return vals

def knn_impute(
    dataset: Dataset, molecule: str, column: str, result_column: Optional[str] = None, **kwargs
) -> Dataset:
    """Apply the scikit learn k nearest neighbor imputation to a dataset.

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
        column (str): Name of the value column to impute.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.
        n_neighbors (int, optional): Number of neighbors to use for imputation. Defaults to 5.
        transpose (bool, optional): Whether to transpose the matrix before imputation. Defaults to False.

    Returns:
        pd.Series: The imputed values.
    """
    imputer = KNNImputer(missing_values=dataset.missing_value, **kwargs)
    imputed = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
        result_column=result_column
    )
    return imputed

def iterative_impute(
    dataset: Dataset, molecule: str, column: str, min_value:Optional[float]=None, max_value:Optional[float]=None,  result_column: Optional[str] = None, **kwargs
) -> pd.Series:
    """Apply the scikit-learn iterative imputer to a dataset that estimates each feature from all the others.

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
        column (str): Name of the value column to impute.
        min_value (Optional[float], optional): Minimum value to impute. If not given set to minimum non-missing value. Defaults to None.
        max_value (Optional[float], optional): Maximum value to impute. If not given set to maximum non-missing value. Defaults to None.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.
        transpose (bool, optional): Whether to transpose the matrix before imputation. Defaults to False.

    Returns:
        pd.Series: The imputed values.
    """
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
        result_column=result_column
    )
    return imputed
