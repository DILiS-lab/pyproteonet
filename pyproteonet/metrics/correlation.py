from typing import Dict, Callable, Optional, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..data.dataset import Dataset


def per_sample_correlation(
    dataset: Dataset,
    molecule: str,
    column: str,
    ground_truth_column: str,
    ignore_missing: bool = False,
    ids: Optional[pd.Index] = None,
    logarithmize: bool = True,
    correlation_measure: Callable = pearsonr,
) -> Dict[str, float]:
    res = {}
    if 'sample' in ids.names:
        second_index = [n for n in ids.names if n!='sample']
        assert len(second_index) == 1
        second_index = second_index[0]
        ids = {sample:ids[second_index] for sample,ids in ids.to_frame().reset_index(drop=True).groupby('sample')}
    else:
        ids = {sample:ids for sample in dataset.sample_names}
    for sample_name, sample in dataset.samples_dict.items():
        values = sample.values[molecule][column]
        gt = sample.values[molecule][ground_truth_column]
        if ignore_missing:
            df = sample.values
            mask = (~df[column].isna()) & (~df[ground_truth_column].isna())
            values = values.loc[mask]
            gt = gt.loc[mask]
        if ids is not None:
            sample_ids = ids[sample_name].loc[ids[sample_name].isin(gt.index)]
            values = values.loc[sample_ids]
            gt = gt.loc[sample_ids]
        if logarithmize:
            values, gt = np.log(values), np.log(gt)
        r, p = correlation_measure(gt, values)
        res[sample_name] = r
    return res


def across_sample_correlation(
    dataset: Dataset,
    molecule: str,
    column: str,
    ground_truth_column: str,
    ignore_missing_values: bool = True,
    molecule_ids: Optional[List] = None,
    logarithmize: bool = True,
    correlation_measure: Callable = pearsonr,
) -> Dict[str, float]:
    res = {}
    if molecule_ids is not None:
        molecule_ids = pd.Series(molecule_ids)
    val, val_mask = dataset.get_column_flat(molecule=molecule, column=column, return_missing_mask=True)
    gt, gt_mask = dataset.get_column_flat(molecule=molecule, column=ground_truth_column, return_missing_mask=True)
    if not ignore_missing_values and (val_mask.sum() > 0 or gt_mask.sum() > 0):
        raise ValueError("There are missing values in the column or ground truth. If you want to ignore them set ignore_missing_values.")
    mask = ~(val_mask | gt_mask)
    val = val[mask]
    gt = gt[mask]
    if molecule_ids is not None:
        sample_gt_mols = gt.index.get_level_values('id').isin(molecule_ids)
        #sample_gt_mols = molecule_ids.loc[sample_gt_mols]
        val = val.loc[sample_gt_mols]
        gt = gt.loc[sample_gt_mols]
    if logarithmize:
        val, gt = np.log(val), np.log(gt)
    r, p = correlation_measure(gt, val)
    return r