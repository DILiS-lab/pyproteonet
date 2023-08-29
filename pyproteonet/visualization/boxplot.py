from typing import List, Callable, Optional, Union

import pandas as pd
from matplotlib import pyplot as plt

from ..data.dataset import Dataset
from ..metrics.compare import compare_columns_with_gt


def plot_per_sample_metric_boxplot(
    dataset: Dataset,
    molecule: str,
    columns: List[str],
    gt_column: str,
    metric: Union[str, Callable] = "pearsonr",
    ignore_missing: bool = True,
    ids: Optional[pd.Index] = None,
    logarithmize: bool = True,
    tick_labels: Optional[List[str]] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
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
    )
    tlabels = []
    for i, column in enumerate(columns):
        tlabel = column if tick_labels is None else tick_labels[i]
        if print_missingness:
            _, missingness = dataset.get_column_flat(molecule=molecule, column=column, return_missing_mask=True)
            missingness = missingness.sum() / missingness.shape[0]
            tlabel += f"\n({missingness * 100:.2f}% missing)"
        tlabels.append(tlabel)
    if ax is None:
        fig, ax = plt.subplots()
    ax.boxplot([list(metric.values()) for metric in metrics.values()])
    ax.set_xticklabels(tlabels, rotation=45)
    ax.set_ylabel(metric)
    ax.set_xlabel("Column")
