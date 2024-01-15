from typing import List, Union, Optional
import itertools
import math

from ..data import Dataset

import pandas as pd
import numpy as np


def calculate_sample_pair_ratios(
    dataset: Dataset,
    molecule: str,
    columns: Union[str, List[str]],
    numerator_samples: List[str],
    denominator_samples: List[str],
    calculate_log2_ratio: bool = False,
    ids: Optional[pd.Index] = None,
    is_log: bool = False
) -> pd.DataFrame:
    """Given a set of numerator and denominator samples, calculate the ratio of the numerator to the denominator for each possible pair of samples.
       Can be used to calculate abundance ratios or fold changes between pairs of two samples.

    Args:
        dataset (Dataset): The dataset to calculate ratios for.
        molecule (str): The molecule type to calculate ratios for.
        columns (Union[str, List[str]]): The value column(s) to calculate ratios for.
        numerator_samples (List[str]): List of samples names to use as nominator when computing ratios.
        denominator_samples (List[str]): List of samples names to use as denominator when computing ratios.
        calculate_log2_ratio (bool, optional): Whether to calculate the log2 ratio instead of the ratio. Defaults to False.
        ids (Optional[pd.Index], optional): If given, only consider molecules with these ids. Defaults to None.
        is_log (bool, optional): Whether the given columns are log transformed and need to be transformed back for ratio calculation. Defaults to False.

    Returns:
        pd.DataFrame: The calculated ratios.
    """
    if isinstance(columns, str):
        columns = [columns]
    vals = dataset.values[molecule].df[columns]
    val_groups = {s: g for s, g in vals.groupby("sample")}
    res = []
    for ns, ds in itertools.product(numerator_samples, denominator_samples):
        if is_log:
            ratio = math.e**(val_groups[ns].droplevel("sample") - val_groups[ds].droplevel("sample"))
        else:
            ratio = val_groups[ns].droplevel("sample") / val_groups[ds].droplevel("sample")
        if calculate_log2_ratio:
            ratio = np.log2(ratio)
        ratio = pd.concat([ratio], keys=[ds], names=["denominator_sample"])
        ratio = pd.concat([ratio], keys=[ns], names=["nominator_sample"])
        res.append(ratio)
    res = pd.concat(res)
    if ids is not None:
        res = res.loc[res.index.get_level_values("id").isin(ids)]
    return res

def calculate_ratio_absolute_error(
    dataset: Dataset,
    molecule: str,
    columns: Union[str, List[str]],
    numerator_samples: List[str],
    denominator_samples: List[str],
    ground_truth_ratios: pd.Series,
    calculate_log2_ratio: bool = False,
    ids: Optional[pd.Index] = None,
    is_log: bool = False,
) -> pd.DataFrame:
    """Calculate the absolute error between the ratios of numerator and denominator samples and a given ground truth for all possible pairs of samples.

    Args:
        dataset (Dataset): The dataset to calculate ratios for.
        molecule (str): The molecule type to calculate ratios for.
        columns (Union[str, List[str]]): The value column(s) to calculate ratios for.
        numerator_samples (List[str]): List of samples names to use as nominator when computing ratios.
        denominator_samples (List[str]): List of samples names to use as denominator when computing ratios.
        ground_truth_ratios (pd.Series): Ground truth ratios to compare the calculated ratios to.
        calculate_log2_ratio (bool, optional): Whether to calculate the log2 ratio instead of the ratio. Defaults to False.
        ids (Optional[pd.Index], optional): If given, only consider molecules with these ids. Defaults to None.
        is_log (bool, optional): Whether the given columns are log transformed and need to be transformed back for ratio calculation. Defaults to False.

    Returns:
        pd.DataFrame: The calculated absolute errors.
    """
    mae = calculate_sample_pair_ratios(
        dataset, molecule, columns, numerator_samples, denominator_samples, ids=ids, is_log=is_log, calculate_log2_ratio=calculate_log2_ratio
    )
    gt = ground_truth_ratios[mae.index.get_level_values('id')].values
    if calculate_log2_ratio:
        gt = np.log2(gt)
    for c in mae.columns:
        mae.loc[:,c] = (mae[c] - gt).abs()
    return mae
