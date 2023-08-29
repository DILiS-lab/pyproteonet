from typing import Dict, Callable, Optional, List, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..data.dataset import Dataset

def _get_metric_from_str(metric: str):
    metric = metric.lower()
    if metric == 'pearsonr':
        res = lambda val, gt: pearsonr(val, gt)[0]
    elif metric == 'spearmanr':
        res = lambda val, gt: spearmanr(val, gt)[0]
    elif metric == 'mse':
        res = lambda val, gt: ((val-gt)**2).mean()
    elif metric == 'rmse':
        res = lambda val, gt: (((val-gt)**2).mean())**0.5
    else:
        raise AttributeError(f"Metric {metric} not found!")
    return res

def compare_columns_with_gt(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    gt_column: str,
    metric: Union[Callable, str] = pearsonr,
    ignore_missing: bool = True,
    ids: Optional[pd.Index] = None,
    logarithmize: bool = True,
    per_sample: bool = False
) -> Dict[str, float]:
    if isinstance(metric, str):
        metric = _get_metric_from_str(metric)
    val = dataset.get_values_flat(molecule=molecule, columns=columns)
    gt = dataset.get_column_flat(molecule=molecule, column=gt_column)
    if ids is not None:
        if 'sample' not in ids.names:
            assert len(ids.names)==1
            concat = []
            for sample_name in dataset.sample_names:
                concat.append(pd.DataFrame({'sample':sample_name, 'id':ids}))
            ids = pd.concat(concat).set_index(('sample', 'id'))
        val = val[val.index.isin(ids)]
        gt = gt[gt.index.isin(ids)]
    gt_missing = gt.isna()
    val_missing = {c:val[c].isna() for c in columns}
    if logarithmize:
        val, gt = np.log(val), np.log(gt)
    if not ignore_missing and (any([vm.sum() > 0 for vm in val_missing.values()]) or gt_missing.sum() > 0):
        raise ValueError("There are missing values in the column or ground truth. If you want to ignore them set ignore_missing_values.")
    else:
        gt_columns = {}
        val_columns = {}
        for c in columns:
            mask = ~(val_missing[c] | gt_missing)
            gt = gt[mask]
            gt_columns[c] = gt
            val_columns[c] =  val.loc[mask, c]
    res = {}
    for c in columns:
        val, gt = val_columns[c], gt_columns[c]
        if per_sample:
            val = {sample:val for sample,val in val.groupby('sample')}
            gt = {sample:gt for sample,gt in gt.groupby('sample')}
            r = {}
            for sample in gt.keys():
                r[sample] = metric(gt[sample], val[sample])
            res[c] = r
        else:
            res[c] = metric(gt, val)
    return res