from typing import Dict, Callable, Optional, List, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..data.dataset import Dataset


def _get_metric_from_str(metric: str):
    metric = metric.lower()
    if metric == "pearsonr":
        res = lambda val, gt: pearsonr(val, gt)[0]
    elif metric == "spearmanr":
        res = lambda val, gt: spearmanr(val, gt)[0]
    elif metric == "mse":
        res = lambda val, gt: ((val - gt) ** 2).mean()
    elif metric == "mae":
        res = lambda val, gt: ((val - gt).abs()).mean()
    elif metric == "rmse":
        res = lambda val, gt: (((val - gt) ** 2).mean()) ** 0.5
    else:
        raise AttributeError(f"Metric {metric} not found!")
    return res


def compare_columns_with_gt(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    gt_column: str,
    ids: Optional[pd.Index] = None,
    metric: Union[Callable, str] = pearsonr,
    ignore_missing: bool = True,
    logarithmize: bool = True,
    per_sample: bool = False,
    return_counts: bool = False,
) -> Dict[str, float]:
    if isinstance(metric, str):
        metric = _get_metric_from_str(metric)
    val = dataset.get_values_flat(molecule=molecule, columns=columns)
    gt = dataset.get_column_flat(molecule=molecule, column=gt_column)
    if ids is not None:
        if "sample" not in ids.names:
            assert len(ids.names) == 1
            concat = []
            for sample_name in dataset.sample_names:
                concat.append(pd.DataFrame({"sample": sample_name, "id": ids}))
            ids = pd.concat(concat).set_index(("sample", "id"))
        val = val[val.index.isin(ids)]
        gt = gt[gt.index.isin(ids)]
    gt_missing = gt.isna()
    val_missing = {c: val[c].isna() for c in columns}
    if logarithmize:
        val, gt = np.log(val), np.log(gt)
    missing_value_columns = [c for c, vm in val_missing.items() if vm.any()]
    if not ignore_missing and (len(missing_value_columns) or gt_missing.any()):
        if len(missing_value_columns):
            raise ValueError(f"There are value columns with missing values: {missing_value_columns}")
        if gt_missing.any():
            raise ValueError(f"The ground truth column {gt_column} has missing values")
    else:
        gt_columns = {}
        val_columns = {}
        for c in columns:
            mask = ~(val_missing[c] | gt_missing)
            gt = gt[mask]
            gt_columns[c] = gt
            val_columns[c] = val.loc[mask, c]
    res = {}
    counts = {}
    for c in columns:
        val, gt = val_columns[c], gt_columns[c]
        if per_sample:
            val = {sample: val for sample, val in val.groupby("sample")}
            gt = {sample: gt for sample, gt in gt.groupby("sample")}
            r = {}
            cnt = {}
            for sample in gt.keys():
                assert gt[sample].shape[0] == val[sample].shape[0]
                cnt[sample] = gt[sample].shape[0]
                r[sample] = metric(gt[sample], val[sample])
            res[c] = r
            counts[c] = cnt
        else:
            assert gt.shape[0] == val.shape[0]
            counts[c] = gt.shape[0]
            res[c] = metric(gt, val)
    if return_counts:
        return res, counts
    else:
        return res


def compare_columns_with_gt_multi_datasets(
    datasets: List[Dataset],
    columns: List[str],
    gt_column: str,
    ids: Optional[List[pd.Index]] = None,
    groups: Optional[List] = None,
    per_sample: bool = False,
    return_counts: bool = False,
    *args,
    **kwargs,
) -> pd.DataFrame:
    if "dataset" in columns:
        raise RuntimeError(
            "The name 'dataset' is reserved for the output dataframe and cannot be the name of a column to compare"
        )
    if "group" in columns and groups is not None:
        raise RuntimeError(
            "The name 'group' is reserved for the output dataframe and cannot be the name of a column to compare"
        )
    if groups is not None:
        if len(groups) != len(datasets):
            raise AttributeError("Lists datasets and groups must have same length")
    if ids is not None:
        if len(ids) != len(datasets):
            raise AttributeError("Lists of ids indices and datasets must have same length")
    metrics_df = []
    counts_df = []
    for i, (dataset, ds_ids) in enumerate(zip(datasets, ids)):
        metrics, counts = compare_columns_with_gt(
            dataset=dataset,
            columns=columns,
            gt_column=gt_column,
            ids=ds_ids,
            per_sample=per_sample,
            return_counts=True,
            *args,
            **kwargs,
        )
        if per_sample:
            metrics = {column: list(metric.values()) for column, metric in metrics.items()}
            counts = {column: list(counts.values()) for column, counts in counts.items()}
        else:
            metrics = {column: [metric] for column, metric in metrics.items()}
            counts = {column: [counts] for column, counts in counts.items()}
        df = pd.DataFrame(metrics)
        df["dataset"] = i
        if groups is not None:
            df["group"] = groups[i]
        metrics_df.append(df)
        counts_df.append(pd.DataFrame(counts))
    metrics_df = pd.concat(metrics_df, ignore_index=True, sort=False)
    if return_counts:
        counts_df = pd.concat(counts_df, ignore_index=True, sort=False)
        return metrics_df, counts_df
    else:
        return metrics_df
