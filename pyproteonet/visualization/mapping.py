from typing import Optional

import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt

from ..data import Dataset

def plot_mapping_degree_distribution(dataset: Dataset, molecule: str, mapping: str, cutoff: int = 30, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,3))
    degs = dataset.molecule_set.get_mapping_degrees(molecule=molecule, mapping=mapping, only_unique=False)
    degs_unique = dataset.molecule_set.get_mapping_degrees(molecule=molecule, mapping=mapping, only_unique=True)
    degs_plot = pd.DataFrame({'All Peptides':degs, 'Unique Peptides':degs_unique})
    degs_plot = degs_plot.stack()
    degs_cutoff = degs_plot[degs_plot >= cutoff]
    degs_plot = degs_plot[degs_plot<cutoff]
    #degs_cutoff = degs_plot[degs_plot >= cutoff].count()
    degs_plot = degs_plot.groupby([degs_plot, degs_plot.index.get_level_values(-1)]).count()
    degs_cutoff = degs_cutoff.groupby(degs_cutoff.index.get_level_values(-1)).count()
    degs_plot = degs_plot.reset_index()
    degs_plot = degs_plot.rename(columns={'level_0':'Degree', 'level_1': 'Peptide Type', 0:'Count'})
    degs_cutoff = degs_cutoff.reset_index()
    degs_cutoff['Degree'] = f'{cutoff}+'
    degs_cutoff.rename(columns={'index':'Peptide Type', 0:'Count'}, inplace=True)
    degs_plot = pd.concat([degs_plot, degs_cutoff], ignore_index=True)
    degs_plot['Percentage'] = degs_plot['Count'] / dataset.molecules[molecule].shape[0] * 100
    sbn.barplot(degs_plot, x='Degree', y='Percentage', hue='Peptide Type', ax=ax)