from typing import List, Optional, Iterable, Union, Tuple
from warnings import warn

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ..data.dataset import Dataset


def plot_ratios_volcano(
    dataset: Dataset,
    molecule: str,
    column: str,
    samples_a: List[str],
    samples_b: List[str],
    molecule_ids: Optional[Iterable] = None,
    normalize: bool = False,
    min_samples: int = 1,
    log_base: int = 10,
    ratio_log_base: int = 2,
    ax: Optional[plt.Axes] = None,
    scatter_color=None,
    plot_missing: bool = True,
    return_mean: bool = False,
):
    values = dataset.get_samples_value_matrix(molecule=molecule, column=column, samples=samples_a + samples_b)
    overall_missing = values.to_numpy().flatten()
    overall_missing = np.isnan(overall_missing).sum() / overall_missing.shape[0]
    mean_sum = values.values.sum() / len(values.columns)
    groups = []
    for sample_group in [samples_a, samples_b]:
        sample_group = values.loc[:, sample_group]
        if normalize:
            sample_group *= sample_group.sum() / mean_sum
        if molecule_ids is not None:
            sample_group = sample_group.loc[molecule_ids]
        sample_group = sample_group.stack(dropna=False)
        sample_group.index.rename(names="id", level=0, inplace=True)
        grouped = sample_group.groupby("id")
        group_mean = grouped.mean()
        non_na = (~sample_group.isna()).groupby("id").sum()
        not_enough_evidence = non_na[~(non_na >= min_samples)].index
        group_mean.loc[not_enough_evidence] = np.nan
        groups.append(group_mean)
    if ax is None:
        fig, ax = plt.subplots()
    abundances = pd.DataFrame({"a": groups[0], "b": groups[1]})
    ratio = abundances.a / abundances.b
    averaged_missing = ratio.isna().sum() / ratio.shape[0]
    mean_abundances = abundances.mean(axis=1)
    if log_base is not None:
        mean_abundances = np.log2(mean_abundances) / np.log2(log_base)
    if ratio_log_base is not None:
        ratio = np.log2(ratio) / np.log2(ratio_log_base)
    ax.scatter(ratio, mean_abundances, color=scatter_color)
    ratio_mean = ratio.mean()
    ratio_existing_abundance = mean_abundances[~ratio.isna()]
    ax.plot(
        [ratio_mean, ratio_mean], [ratio_existing_abundance.min(), ratio_existing_abundance.max()], color=scatter_color
    )
    if plot_missing:
        missing_text = (
            f"{round(averaged_missing*100,2)}% ratios missing \n({round(overall_missing*100,2)}% of all values missing)"
        )
        ax.annotate(
            missing_text,
            xy=(0, 1),
            xytext=(12, -12),
            va="top",
            xycoords="axes fraction",
            textcoords="offset points",
            color="red",
        )
    ax.text(
        ratio_mean,
        ratio_existing_abundance.min(),
        f"ratio mean: {round(ratio.mean(), 2)}",
        horizontalalignment="center",
        verticalalignment="top",
        size=11,
    )
    ax.set_xlabel("Ratio" + f" (log{str(ratio_log_base)})" if ratio_log_base is not None else "")
    ax.set_ylabel("Mean Abundance" + f" (log{str(log_base)})" if log_base is not None else "")
    if return_mean:
        return ratio_mean


def plot_ratios_difference_volcano(
    dataset: Dataset,
    molecule: str,
    column: str,
    samples_a: List[str],
    samples_b: List[str],
    molecule_ids: List[Iterable],
    normalize: bool = False,
    min_samples: int = 1,
    log_base: int = 10,
    ratio_log_base: int = 2,
    ax: Optional[plt.Axes] = None,
    scatter_colors: List[Union[str, Tuple[float, float ,float]]] = ["orange", "green"],
):
    if len(molecule_ids) != len(scatter_colors):
        warn(
            "List of molecule id sequences has different length than list of scatter_colors"
            + " used to plot those molecule ratios, using default color for all ratios!"
        )
        scatter_colors = [None] * len(molecule_ids)
    if ax is None:
        fig, ax = plt.subplots()
    means = []
    for ids, color in zip(molecule_ids, scatter_colors):
        mean = plot_ratios_volcano(
            dataset=dataset,
            molecule=molecule,
            column=column,
            samples_a=samples_a,
            samples_b=samples_b,
            molecule_ids=ids,
            normalize=normalize,
            min_samples=min_samples,
            log_base=log_base,
            ratio_log_base=ratio_log_base,
            ax=ax,
            scatter_color=color,
            plot_missing=False,
            return_mean=True,
        )
        means.append(mean)
    for i in range(len(means) - 1):
        start = (means[i], 0.9)
        end = (means[i + 1], 0.9)
        ax.annotate("", xy=start, xytext=end, arrowprops=dict(arrowstyle="<->"), xycoords=("data", "axes fraction"))
        ax.annotate(
            str(round(means[i + 1] - means[i], 2)),
            xy=((means[i + 1] + means[i]) / 2, 0.9),
            size=11,
            xycoords=("data", "axes fraction"),
            horizontalalignment="center",
        )
