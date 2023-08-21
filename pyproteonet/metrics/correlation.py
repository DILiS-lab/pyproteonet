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
    ignore_ground_truth_missing_values: bool = False,
    ground_truth_molecules: Optional[List] = None,
    logarithmize: bool = True,
    correlation_measure: Callable = pearsonr,
) -> Dict[str, float]:
    res = {}
    if ground_truth_molecules is not None:
        ground_truth_molecules = pd.Series(ground_truth_molecules)
    for sample_name, sample in dataset.samples_dict.items():
        mask = sample.non_missing_mask(molecule=molecule, column=column)
        if ignore_ground_truth_missing_values:
            mask = mask & sample.non_missing_mask(molecule=molecule, column=ground_truth_column)
        values = sample.values[molecule].loc[mask, column]
        gt = sample.values[molecule].loc[mask, ground_truth_column]
        if ground_truth_molecules is not None:
            sample_gt_mols = ground_truth_molecules.isin(gt.index)
            sample_gt_mols = ground_truth_molecules.loc[sample_gt_mols]
            values = values.loc[sample_gt_mols]
            gt = gt.loc[sample_gt_mols]
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