from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.dataset import Dataset


def normalize_sum(
    dataset: Dataset,
    molecule: str,
    column: str,
    reference_ids: Optional[pd.Index] = None,
    reference_sample: Optional[str] = None,
    result_column: Optional[str] = None,
):
    if reference_sample is None:
        reference_sample = list(dataset.sample_names)[0]
    values = dataset.get_column_flat(molecule=molecule, column=column, ids=reference_ids)
    factors = values.groupby("sample").sum()
    factors = factors[reference_sample] / factors
    values = dataset.get_column_flat(molecule=molecule, column=column)
    factors = values.index.get_level_values("sample").map(factors)
    values = values * factors
    if result_column is not None:
        dataset.values[molecule][result_column] = values
    return values
