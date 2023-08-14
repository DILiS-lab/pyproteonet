from typing import List, Callable, Optional, Union
import itertools

import numpy as np
from matplotlib import pyplot as plt

from ..data.dataset import Dataset


def plot_error_boxplot(
    dataset: Dataset,
    columns: List[str],
    ground_truth: Union[str, List[str]] = "abundance",
    molecule: str = "protein",
    logarithmize: bool = True,
    error_measure: str = "rmse",
    tick_labels: Optional[List[str]] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
    errors = []
    tlabels = []
    gt_columns = len(columns) * [ground_truth] if isinstance(ground_truth, str) else ground_truth
    if len(gt_columns) != len(columns):
        raise ValueError(
            f"ground_truth must be a string or a list of strings with the same size as the columns parameter"
        )
    error_measure = error_measure.lower()
    if error_measure == "mse":
        error_function = lambda x, gt: ((x - gt) ** 2).mean()
    elif error_measure == "rmse":
        error_function = lambda x, gt: np.sqrt((x - gt) ** 2).mean()
    else:
        raise KeyError("Parameter error_measur must be in mse or rmse!")
    for i, column in enumerate(columns):
        num_values = 0
        num_missing = 0
        column_errors = []
        for sample in dataset.samples:
            mask = sample.non_missing_mask(molecule=molecule, column=column)
            num_values += mask.shape[0]
            num_missing += (~mask).sum()
            values = sample.values[molecule].loc[mask, column]
            gt = sample.values[molecule].loc[mask, gt_columns[i]]
            if logarithmize:
                values, gt = np.log(values), np.log(gt)
            error = error_function(values, gt)
            column_errors.append(error)
        tlabel = column if tick_labels is None else tick_labels[i]
        if print_missingness:
            tlabel += f"\n({round(num_missing/num_values, 2) * 100}% missing)"
        tlabels.append(tlabel)
        errors.append(column_errors)
    if ax is None:
        fig, ax = plt.subplots()
    ax.boxplot(errors)
    ax.set_xticklabels(tlabels, rotation=45)
    ax.set_ylabel(f"{error_measure.upper()} Error{' (on logarithmized values)' if logarithmize else ''}")
    ax.set_xlabel("Column")


def plot_ratio_error_boxplot(
    dataset: Dataset,
    columns: List[str],
    samples_a: List[str],
    samples_b: List[str],
    ground_truth_ratio: float,
    molecule: str = "protein",
    ratio_log_base: Optional[int] = 2,
    error_measure: str = "rmse",
    average_ratios: bool = False,
    tick_labels: Optional[List[str]] = None,
    print_missingness: bool = True,
    ax: Optional[plt.Axes] = None,
):
    errors = []
    tlabels = []
    error_measure = error_measure.lower()
    if error_measure == "mse":
        error_function = lambda x, gt: ((x - gt) ** 2).mean()
    elif error_measure == "rmse":
        error_function = lambda x, gt: np.sqrt((x - gt) ** 2).mean()
    else:
        raise KeyError("Parameter error_measur must be in mse or rmse!")
    sample_pairs = list(itertools.product(samples_a, samples_b))
    for i, column in enumerate(columns):
        column_errors = []
        for a, b in sample_pairs:
            a = dataset[a].values[molecule][column]
            b = dataset[b].values[molecule][column]
            ratio = a / b
            if average_ratios:
                ratio = np.array([ratio.mean()])
            if ratio_log_base is not None:
                ratio = np.log2(ratio) / np.log2(ratio_log_base)
                error = error_function(ratio, np.log2(ground_truth_ratio) / np.log2(ratio_log_base))
            else:
                error = error_function(ratio, ground_truth_ratio)
            column_errors.append(error)
        tlabel = column if tick_labels is None else tick_labels[i]
        if print_missingness:
            vals, mask = dataset.get_column_flat(molecule=molecule, column=column, return_missing_mask=True)
            tlabel += f"\n({round(mask.sum() / vals.shape[0], 2) * 100}% missing)"
        tlabels.append(tlabel)
        errors.append(column_errors)
    if ax is None:
        fig, ax = plt.subplots()
    ax.boxplot(errors)
    ax.set_xticklabels(tlabels, rotation=45)
    ax.set_ylabel(
        f"{error_measure.upper()} Ratio Error{f' (on log{ratio_log_base} ratios)' if ratio_log_base is not None else ''}"
    )
    ax.set_xlabel("Column")
