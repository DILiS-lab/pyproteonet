from typing import Dict, Callable, Optional, List, Union, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..data.dataset import Dataset


def _get_metric_from_str(metric: str):
    metric = metric.lower()
    if metric == "pearsonr":
        res = lambda val, gt: [pearsonr(val, gt)[0]]
    elif metric == "spearmanr":
        res = lambda val, gt: [spearmanr(val, gt)[0]]
    elif metric == "mse":
        res = lambda val, gt: [((val - gt) ** 2).mean()]
    elif metric == "mae":
        res = lambda val, gt: [((val - gt).abs()).mean()]
    elif metric == "rmse":
        res = lambda val, gt: [(((val - gt) ** 2).mean()) ** 0.5]
    else:
        raise AttributeError(f"Metric {metric} not found!")
    return res


def _compare_columns_single_dataset(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    comparison_column: str,
    ids: Optional[pd.Index] = None,
    metric: Union[Literal['PearsonR', 'SpearmanR', 'MSE', 'MAE', 'RMSE'], Callable[[pd.Series, pd.Series], pd.Series]] = 'PearsonR',
    ignore_missing: bool = True,
    logarithmize: bool = True,
    per_sample: bool = False,
    return_counts: bool = False,
    replace_nan_metric_with: Optional[float] = None
) -> Dict[str, float]:
    if isinstance(metric, str):
        metric = _get_metric_from_str(metric)
    val = dataset.get_values_flat(molecule=molecule, columns=columns)
    gt = dataset.get_column_flat(molecule=molecule, column=comparison_column)
    if ids is not None:
        if "sample" not in ids.names:
            assert len(ids.names) == 1
            concat = []
            for sample_name in dataset.sample_names:
                concat.append(pd.DataFrame({"sample": sample_name, "id": ids}))
            ids = pd.MultiIndex.from_frame(pd.concat(concat))
        val = val[val.index.isin(ids)]
        gt = gt[gt.index.isin(ids)]
        if len(gt)==0 or len(val) == 0:
            raise ValueError("No values for comparison for the given ids")
    gt_missing = gt.isna()
    val_missing = {c: val[c].isna() for c in columns}
    if logarithmize:
        val, gt = np.log(val), np.log(gt)
    missing_value_columns = [c for c, vm in val_missing.items() if vm.any()]
    if not ignore_missing and (len(missing_value_columns) or gt_missing.any()):
        if len(missing_value_columns):
            raise ValueError(f"There are value columns with missing values: {missing_value_columns}")
        if gt_missing.any():
            raise ValueError(f"The ground truth column {comparison_column} has missing values")
    else:
        gt_columns = {}
        val_columns = {}
        for c in columns:
            mask = ~(val_missing[c] | gt_missing)
            gt_c = gt[mask]
            gt_columns[c] = gt_c
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
                met = metric(gt[sample], val[sample])
                if replace_nan_metric_with is not None and np.isnan(met):
                    met = [replace_nan_metric_with]
                r[sample] = met
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


#TODO: write docstring
def compare_columns(
    dataset: Union[Dataset, List[Dataset]],
    molecule: str,
    columns: List[str],
    comparison_column: str,
    ids: Optional[pd.Index] = None,
    metric: Literal['PearsonR', 'SpearmanR', 'MSE', 'MAE', 'RMSE'] = 'PearsonR',
    ignore_missing: bool = True,
    logarithmize: bool = True,
    per_sample: bool = False,
    replace_nan_metric_with: Optional[float] = None
) -> pd.DataFrame:
    if isinstance(dataset, Dataset):
        dataset = [dataset]
    res_dfs = []
    for i_ds, ds in enumerate(dataset):
        metric, counts = _compare_columns_single_dataset(dataset=ds, molecule=molecule, columns=columns, comparison_column=comparison_column,
                                                         ids=ids, metric=metric, ignore_missing=ignore_missing, logarithmize=logarithmize,
                                                         per_sample=per_sample, return_counts=True, replace_nan_metric_with=replace_nan_metric_with)
        for col, met in metric.items():
            cnt = counts[col]
            if per_sample:
                col_df = []
                for sample_name, metric_value in met.items():
                    df = pd.DataFrame({'metric': metric_value})
                    df['sample'] = sample_name
                    df['count'] = cnt[sample_name]
                    col_df.append(df)
                col_df = pd.concat(col_df, ignore_index=True)
            else:
                col_df = pd.DataFrame({'metric': met})
                col_df['count'] = cnt
            col_df['column'] = col
            if len(dataset) > 1:
                col_df['dataset'] = i_ds
            res_dfs.append(col_df)
    return pd.concat(res_dfs, ignore_index=True)

#TODO: write docstring
def compare_columns_molecule_groups(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    comparison_column: str,
    id_groups: Optional[Dict[str, pd.Index]] = None,
    per_sample: bool = False,
    return_counts: bool = False,
    *args,
    **kwargs,
) -> pd.DataFrame:
    metrics_df = []
    counts_df = []
    if 'group' in columns:
        raise ValueError("Column name 'group' is reserved in the output and can therefore not be part of the columns!")
    for group, group_ids in id_groups.items():
        df = compare_columns(
            dataset=dataset,
            molecule=molecule,
            columns=columns,
            comparison_column=comparison_column,
            ids=group_ids,
            per_sample=per_sample,
            *args,
            **kwargs,
        )
        df["group"] = group
        metrics_df.append(df)
    return pd.concat(metrics_df, ignore_index=True, sort=False)
