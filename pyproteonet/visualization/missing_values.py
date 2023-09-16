from typing import Optional

import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt

from ..data import Dataset

def plot_samples_per_molecule_missing(dataset: Dataset, molecule: str, column: str, ax: Optional[plt.Axes] = None):
    pep_mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    pep_missingness = pep_mat.isna().sum(axis=1)
    pep_missingness = pep_missingness.groupby(pep_missingness).count() / pep_mat.shape[0] * 100
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,3))
    sbn.barplot(x=pep_missingness.index, y=pep_missingness, ax=ax)
    ax.bar_label(ax.containers[0], fmt='%.2f')
    ax.set_xlabel('Number Missing Sample Values per Peptide')
    ax.set_ylabel('Percentage Peptides')
    ax.set_ylim(0, pep_missingness.max() + 10)
    ax.text(pep_mat.shape[1] / 2, pep_missingness.max() + 5, horizontalalignment='center',
            s=f'Overall Percentage Missing Values: {pep_mat.isna().to_numpy().sum() * 100 / pep_mat.shape[0] / pep_mat.shape[1]:.2f}%')