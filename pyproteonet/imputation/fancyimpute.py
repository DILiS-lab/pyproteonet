from typing import Optional, Callable

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
    inplace: bool = False,
    **kwargs
) -> Dataset:
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    matrix = dataset.get_samples_value_matrix(molecule=molecule, value_column=column)
    matrix_vals = matrix.values
    missing_mask = eq_nan(matrix_vals, dataset.missing_value)
    matrix_vals[missing_mask] = 0
    matrix_imputed = imputer.solve(matrix_vals, missing_mask=missing_mask)
    matrix_imputed = pd.DataFrame(matrix_imputed, columns=matrix.columns, index=matrix.index)
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals

def iterative_svd_impute(
    dataset: Dataset, molecule: str, column: str, result_column: Optional[str] = None, inplace: bool = False, **kwargs
) -> Dataset:
    imputer = IterativeSVD(**kwargs)
    dataset = generic_fancy_impute(
        dataset=dataset, molecule=molecule, column=column, imputer=imputer, result_column=result_column, inplace=inplace
    )
    return dataset
