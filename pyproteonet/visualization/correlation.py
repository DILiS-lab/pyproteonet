from typing import List, Callable, Optional, Union

import numpy as np
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

from ..data.dataset import Dataset

def plot_correlation_boxplot(
    dataset: Dataset,
    columns: List[str],
    ground_truth: Union[str, List[str]] = "abundance",
    molecule: str = "protein",
    logarithmize: bool = True,
    correlation_measure: Callable = pearsonr,
    tick_labels: Optional[List[str]] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
    correlations = []
    tlabels = []
    gt_columns = len(columns) * [ground_truth] if isinstance(ground_truth, str) else ground_truth
    if len(gt_columns) != len(columns):
        raise ValueError(f"ground_truth must be a string or a list of strings with the same size as the columns parameter")
    for i, column in enumerate(columns):
        num_values = 0
        num_missing = 0
        column_correlations = []
        for sample in dataset.samples:
            mask = sample.non_missing_mask(molecule=molecule, column=column)
            num_values += mask.shape[0]
            num_missing += (~mask).sum()
            values = sample.values[molecule].loc[mask, column]
            gt = sample.values[molecule].loc[mask, gt_columns[i]]
            if logarithmize:
                values, gt = np.log(values), np.log(gt)
            r, p = correlation_measure(gt, values)
            column_correlations.append(r)
        tlabel = column if tick_labels is None else tick_labels[i]
        if print_missingness:
            tlabel += f"\n({round(num_missing/num_values, 2) * 100}% missing)"
        tlabels.append(tlabel)
        correlations.append(column_correlations)
    if ax is None:
        fig, ax = plt.subplots()
    ax.boxplot(correlations)
    ax.set_xticklabels(tlabels, rotation=45)
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Column")
