from typing import Optional, Callable
import sys

import numpy as np
import pandas as pd
import sklearn

from .missingpy import MissForest
from ..data.dataset import Dataset
from .sklearn import generic_matrix_imputation

def missing_forrest_impute(
    dataset: Dataset, molecule: str, column: str, **kwargs
) -> Dataset:
    imputer = MissForest(missing_values=dataset.missing_value, **kwargs)
    dataset = generic_matrix_imputation(
        dataset=dataset,
        molecule=molecule,
        column=column,
        imputation_function=imputer.fit_transform,
    )
    return dataset
