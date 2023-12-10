from typing import List, Union, Optional
import itertools

from ..data import Dataset

import pandas as pd
import numpy as np


# TODO: Write Docstring
def calculate_sample_pair_ratios(
    dataset: Dataset,
    molecule: str,
    columns: Union[str, List[str]],
    numerator_samples: List[str],
    denominator_samples: List[str],
    ids: Optional[pd.Index] = None,
    is_log: bool = False
) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    vals = dataset.values[molecule].df
    val_groups = {s: g for s, g in vals.groupby("sample")}
    res = []
    for ns, ds in itertools.product(numerator_samples, denominator_samples):
        if is_log:
            ratio = val_groups[ns].droplevel("sample") - val_groups[ds].droplevel("sample")
        else:
            ratio = val_groups[ns].droplevel("sample") / val_groups[ds].droplevel("sample")
        ratio = pd.concat([ratio], keys=[ds], names=["denominator_sample"])
        ratio = pd.concat([ratio], keys=[ns], names=["nominator_sample"])
        res.append(ratio)
    res = pd.concat(res)
    if ids is not None:
        res = res.loc[res.index.get_level_values("id").isin(ids)]
    return res

def caclulate_ratio_absolute_error(
    dataset: Dataset,
    molecule: str,
    columns: Union[str, List[str]],
    numerator_samples: List[str],
    denominator_samples: List[str],
    ground_truth_ratios: pd.Series,
    ids: Optional[pd.Index] = None,
    is_log: bool = False,
) -> pd.DataFrame:
    mae = calculate_sample_pair_ratios(
        dataset, molecule, columns, numerator_samples, denominator_samples, ids=ids, is_log=is_log
    )
    gt = ground_truth_ratios[mae.index.get_level_values('id')].values
    if is_log:
        gt = np.log(gt)
    for c in mae.columns:
        mae.loc[:,c] = (mae[c] - gt).abs()
    return mae
