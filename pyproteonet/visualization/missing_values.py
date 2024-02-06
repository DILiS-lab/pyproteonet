from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ..data import Dataset

def plot_missing_distribution(dataset: Dataset, molecule: str, column: str, ax: Optional[plt.Axes] = None):
    pep_mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    pep_missingness = pep_mat.isna().sum(axis=1)
    pep_missingness = pep_missingness.groupby(pep_missingness).count() / pep_mat.shape[0] * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,3))
    sns.barplot(x=pep_missingness.index, y=pep_missingness, ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.2f', rotation=90)
    ax.set_xlabel('Number Missing Sample Values per Molecule')
    ax.set_ylabel('%')
    ax.set_ylim(0, pep_missingness.max() + 10)
    ax.text(pep_mat.shape[1] / 2, pep_missingness.max() + 5, horizontalalignment='center',
            s=f'Overall Percentage Missing Values: {pep_mat.isna().to_numpy().sum() * 100 / pep_mat.shape[0] / pep_mat.shape[1]:.2f}%')
    
def plot_missingness_categories(dataset: Dataset, molecule: str, column: str, logarithmize: bool = True, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    vals = dataset.values[molecule][column]
    log_means = vals.groupby('id').mean()
    if logarithmize:
        log_means = np.log(log_means)
    missing = vals.isna().groupby('id').sum()
    plot_data = pd.DataFrame({'missing samples':missing, 'log_mean':log_means})
    sns.violinplot(data=plot_data, x='missing samples', y='log_mean', ax=ax)
    ax.set_xlabel('Number Missing')
    ax.set_ylabel(column)
