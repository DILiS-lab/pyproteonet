from typing import List, Callable, Optional, Union, Dict, Literal
import math

import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from ..data.dataset import Dataset
from ..metrics.compare import compare_columns_with_gt, compare_columns_with_gt_multi_datasets


def plot_per_sample_metric_boxplot(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    gt_column: str,
    metric: Union[str, Callable] = "pearsonr",
    ignore_missing: bool = True,
    ids: Optional[pd.Index] = None,
    logarithmize: bool = True,
    replace_nan_metric_with: Optional[float] = None,
    tick_labels: Optional[List[str]] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
    if ax is None:
        fig, ax = plt.subplots()
    metrics = compare_columns_with_gt(
        dataset=dataset,
        molecule=molecule,
        columns=columns,
        gt_column=gt_column,
        metric=metric,
        ignore_missing=ignore_missing,
        ids=ids,
        logarithmize=logarithmize,
        per_sample=True,
        replace_nan_metric_with=replace_nan_metric_with,
    )
    tlabels = []
    for i, column in enumerate(columns):
        tlabel = column if tick_labels is None else tick_labels[i]
        if print_missingness:
            _, missingness = dataset.get_column_flat(molecule=molecule, column=column, return_missing_mask=True)
            missingness = missingness.sum() / missingness.shape[0]
            tlabel += f"\n({missingness * 100:.2f}% missing)"
        tlabels.append(tlabel)
    ax.boxplot([list(metric.values()) for metric in metrics.values()])
    ax.set_xticklabels(tlabels, rotation=45)
    ax.set_ylabel(metric)
    ax.set_xlabel("Column")


def across_datasets_metric_boxplot(
    datasets: List[Dataset],
    molecule: str,
    columns: List[str],
    gt_column: str,
    metric: Union[str, Callable] = "pearsonr",
    ignore_missing: bool = True,
    groups: Optional[List] = None, 
    ids: Optional[List[pd.Index]] = None,
    per_sample: bool = True,
    logarithmize: bool = True,
    replace_nan_metric_with: Optional[float] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
    if len(datasets) != len(ids):
        raise ValueError("List of datases and List of indexes (ids) need to have same length!")
    if ax is None:
        fig, ax = plt.subplots()
    column_missingness = dict()
    metrics_df = compare_columns_with_gt_multi_datasets(
        datasets=datasets,
        molecule=molecule,
        columns=columns,
        gt_column=gt_column,
        metric=metric,
        ignore_missing=ignore_missing,
        ids=ids,
        groups = groups,
        logarithmize=logarithmize,
        per_sample=per_sample,
        replace_nan_metric_with=replace_nan_metric_with
    )
    del metrics_df['dataset']
    tlabels = []
    if print_missingness:
        if groups is not None:
            raise AttributeError('Printing Missingness is not supported when plotting groups')
        for i, (dataset, id) in enumerate(zip(datasets, ids)):
            for i, column in enumerate(columns):
                if print_missingness:
                    _, missingness = dataset.get_column_flat(molecule=molecule, column=column, return_missing_mask=True)
                    cm = column_missingness.get(column, (0, 0))
                    cm = (cm[0] + missingness.sum(), cm[1] + missingness.shape[0])
                    column_missingness[column] = cm
        for column in metrics_df.columns:
            missingness = column_missingness[column]
            missingness = missingness[0] / missingness[1]
            tlabel += f"\n({missingness * 100:.2f}% missing)"
            tlabels.append(tlabel)
    else:
        tlabels = metrics_df.columns
    #ax.boxplot([list(metrics_df[column]) for column in metrics_df.columns])
    if groups is None:
        sns.boxplot(data=metrics_df, ax=ax)
    else:
        metrics_df = pd.melt(metrics_df, id_vars=['group'],value_vars=columns, var_name='Column')
        y='value'
        if isinstance(metric, str):
            metrics_df.rename(columns={'value':metric}, inplace=True)
            y = metric
        sns.boxplot(data=metrics_df, x='group', y=y, hue='Column', ax=ax)
    #ax.set_xticklabels(tlabels, rotation=45)
    #ax.set_ylabel(metric)
    #ax.set_xlabel("Column")

def across_datasets_metric_multi_boxplot(
    datasets: List[Dataset],
    molecule: str,
    columns: List[str],
    gt_columns: Union[str, Dict[str, str]],
    ids: Dict[str, List[pd.Index]] = None,
    axs: Optional[plt.Axes] = None,
    **kwargs
):
    if axs is None:
        nrows = int(math.ceil(len(ids) / 3))
        fig, axs = plt.subplots(nrows, 3, figsize=(12, nrows * 2))
        axs = axs.flatten()
    if isinstance(gt_columns, str):
        gt_columns = {key:gt_columns for key in ids.keys()}
    for ax, (key, ids_list) in zip(axs, ids.items()):
        if not isinstance(ids_list, list):
            ids_list = [ids_list] * len(datasets)
        across_datasets_metric_boxplot(datasets=datasets, molecule=molecule, columns=columns, gt_column=gt_columns[key], ids=ids_list, ax=ax, **kwargs)

def across_datasets_metric_multi_boxplot_grouped(
    datasets: List[Dataset],
    molecule: str,
    columns: List[str],
    gt_columns: Union[str, Dict[str, str]],
    ids: Dict[str, List[pd.Index]] = None,
    groups: Optional[List] = None,
    per_sample: bool = True,
    logarithmize: bool = True,
    ignore_missing: bool = False,
    metric: Union[str, Callable] = 'PearsonR',
    replace_nan_metric_with: Optional[float] = None,
    axs: Optional[plt.Axes] = None,
    axes_level: Literal['group', 'ids', 'column'] = 'ids',
    group_level: Literal['group', 'ids', 'column'] = 'group',
    hue_level: Literal['group', 'ids', 'column'] = 'column',
):
    if "counts" in columns:
        raise RuntimeError(
            "The name 'counts' is reserved for the output dataframe and cannot be the name of a column to compare"
        )
    if len(datasets) != len(groups):
        raise AttributeError("Datasets and groups must have same length")
    if {axes_level, group_level, hue_level} != {'group', 'ids', 'column'}:
        raise AttributeError("Axis/Group/Hue level must be one of ['group', 'ids', 'column'].")
    if len({axes_level, group_level, hue_level}) != 3:
        raise AttributeError("Axis/Group/Hue level must be different.")
    if axes_level == 'group':
        num_plots = len(set(groups))
    elif axes_level == 'ids':
        num_plots = len(ids)
    elif axes_level == 'column':
        num_plots = len(columns)
    if axs is None:
        nrows = int(math.ceil(len(num_plots) / 3))
        fig, axs = plt.subplots(nrows, 3, figsize=(12, nrows * 2))
        axs = axs.flatten()
    if len(axs) != num_plots:
        raise AttributeError(f"Length of axs should be {num_plots} but is {len(axs)}")
    if isinstance(gt_columns, str):
        gt_columns = {key:gt_columns for key in ids.keys()}
    metrics_df = []
    for key, ids_list in ids.items():
        print(key)
        if not isinstance(ids_list, list):
            ids_list = [ids_list] * len(datasets)
        if len(datasets) != len(ids_list):
            raise ValueError(f"List of datases and List of indexes (ids) for key {key} do not have same length!")
        column_missingness = dict()
        metrics_df_ids, counts_df = compare_columns_with_gt_multi_datasets(
            datasets=datasets,
            molecule=molecule,
            columns=columns,
            gt_column=gt_columns[key],
            metric=metric,
            ids=ids_list,
            groups = groups,
            per_sample=per_sample,
            logarithmize=logarithmize,
            ignore_missing=ignore_missing,
            replace_nan_metric_with=replace_nan_metric_with,
            return_counts=True
        )
        metrics_df_ids['ids'] = key
        metrics_df_ids = pd.melt(metrics_df_ids, id_vars=['ids', 'dataset', 'group'], value_vars=columns, var_name='column', ignore_index=True)
        counts_df = pd.melt(counts_df, id_vars=None, value_vars=columns, var_name='column', ignore_index=True)
        metrics_df_ids['count'] = counts_df['value']
        metrics_df.append(metrics_df_ids)
        
    metrics_df = pd.concat(metrics_df)
    metrics_df_grouped = {key: df for key,df in metrics_df.groupby(axes_level, sort=False)}
    for ax, (key, metrics_df_group) in zip(axs, metrics_df_grouped.items()):
        del metrics_df_group[axes_level]
        y='value'
        if isinstance(metric, str):
            metrics_df_group.rename(columns={'value':metric}, inplace=True)
            y = metric
        sns.boxplot(data=metrics_df_group, x=group_level, y=y, hue=hue_level, ax=ax)
        _boxplot_fill_color_to_line_color(ax=ax)
        ax.grid(which='both', axis='y')
        ax.set_title(key)
    return metrics_df

def _boxplot_fill_color_to_line_color(ax):
    box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
    if len(box_patches) == 0:  # in matplotlib older than 3.5, the boxes are stored in ax2.artists
        box_patches = ax.artists
    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()
        patch.set_edgecolor(col)
        patch.set_facecolor('None')

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
            line.set_color(col)
            line.set_mfc(col)  # facecolor of fliers
            line.set_mec(col)  # edgecolor of fliers