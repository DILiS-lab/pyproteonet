from typing import Optional, Callable

import numpy as np
import pandas as pd
from missingpy import MissForrest

from ..data.dataset import Dataset
from .sklearn import generic_matrix_imputation

def missing_forrest_impute(
    dataset: Dataset, molecule: str, column: str, result_column: Optional[str] = None, inplace: bool = False, **kwargs
) -> Dataset:
    imputer = MissForrest(missing_values=dataset.missing_value, **kwargs)
    dataset = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
        result_column=result_column,
        inplace=inplace,
    )
    return dataset
