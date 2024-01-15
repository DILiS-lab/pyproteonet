from typing import Optional, Union
import math

import numpy as np
import pandas as pd
from fancyimpute import IterativeSVD, Solver

from ..utils.numpy import eq_nan
from ..data.dataset import Dataset


def generic_fancy_impute(
    dataset: Dataset,
    molecule: str,
    column: str,
    imputer: Solver,
    result_column: Optional[str] = None,
    **kwargs
) -> pd.Series:
    """Apply a generic fancyimpute function to impute missing values in a dataset.

    Args:
        dataset (Dataset): Input dataset.
        molecule (str): Molecule type to impute.
        column (str): Name of value column to impute.
        imputer (Solver): Name of fancyimpute solver to use.
        result_column (Optional[str], optional): Value column to store imputed values in. Defaults to None.

    Returns:
        pd.Series: imputed values
    """
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    matrix_vals = matrix.values
    missing_mask = eq_nan(matrix_vals, dataset.missing_value)
    matrix_vals[missing_mask] = 0
    matrix_imputed = imputer.solve(matrix_vals, missing_mask=missing_mask)
    print(matrix_imputed.mean())
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns, index=matrix.index)
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    if result_column is not None:
        dataset.values[molecule][result_column] = vals
    return vals


def iterative_svd_impute(
    dataset: Dataset,
    molecule: str,
    column: str,
    result_column: Optional[str] = None,
    min_value: Optional[float] = None,
    rank: Union[float, int] = 0.2,
    **kwargs
) -> Dataset:
    """Impute missing values using iterative singular value decomposition.

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecular type to impute.
        column (str): Value column to impute.
        result_column (Optional[str], optional): If given imputed results are stored under this name. Defaults to None.
        min_value (Optional[float], optional): If given imputation are restricted by this value, otherwise the minimum of the dataset is used. Defaults to None.
        rank (Union[float, int], optional): Rank of the low dimensional representation, either given as integer or float <= 1.0 representing the fraction of number of samples used as rank. Defaults to 0.2.

    Returns:
        pd.Series: imputed values
    """
    if min_value is None:
        min_value = dataset.values[molecule][column].min()
    if rank == -1:
        rank = len(dataset.sample_names) - 1
    if 0 < rank < 1.0:
        rank = min(math.ceil(len(dataset.sample_names) * 0.2), len(dataset.sample_names) - 1)
    imputer = IterativeSVD(min_value=min_value, rank=rank, **kwargs)
    res = generic_fancy_impute(
        dataset=dataset, molecule=molecule, column=column, imputer=imputer, result_column=result_column,
    )
    return res
