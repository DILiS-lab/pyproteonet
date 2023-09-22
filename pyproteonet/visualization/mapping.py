from typing import Optional
import math

import pandas as pd
import seaborn as sbn
from matplotlib import pyplot as plt

from ..data import Dataset

def plot_mapping_degree_distribution(dataset: Dataset, molecule: str, mapping: str,
                                     cutoff: int = 30, cut_top_k: Optional[int] = None, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,3))
    degs = dataset.molecule_set.get_mapping_degrees(molecule=molecule, mapping=mapping, only_unique=False)
    degs_unique = dataset.molecule_set.get_mapping_degrees(molecule=molecule, mapping=mapping, only_unique=True)
    degs_shared = degs - degs_unique
    degs_plot = pd.DataFrame({'All':degs, 'Unique':degs_unique, 'Shared': degs_shared})
    degs_plot = degs_plot.stack()
    degs_cutoff = degs_plot[degs_plot >= cutoff]
    degs_plot = degs_plot[degs_plot<cutoff]
    #degs_cutoff = degs_plot[degs_plot >= cutoff].count()
    degs_plot = degs_plot.groupby([degs_plot, degs_plot.index.get_level_values(-1)]).count()
    degs_cutoff = degs_cutoff.groupby(degs_cutoff.index.get_level_values(-1)).count()
    degs_plot = degs_plot.reset_index()
    degs_plot = degs_plot.rename(columns={'level_0':'Degree', 'level_1': 'Type', 0:'Count'})
    degs_cutoff = degs_cutoff.reset_index()
    degs_cutoff['Degree'] = f'{cutoff}+'
    degs_cutoff.rename(columns={'index':f'Type', 0:'Count'}, inplace=True)
    degs_plot = pd.concat([degs_plot, degs_cutoff], ignore_index=True)
    degs_plot['Percentage'] = degs_plot['Count'] / dataset.molecules[molecule].shape[0] * 100
    topk_height = math.inf
    if cut_top_k:
        topk_height = degs_plot.Percentage.nlargest(n=cut_top_k+1).min() + 10
        ax.set_ylim(0, topk_height)
    sbn.barplot(degs_plot, x='Degree', y='Percentage', hue='Type', ax=ax, hue_order=['Shared', 'Unique', 'All'])
    for container in ax.containers:
        heights = []
        for bar in container:
            h = bar.get_height()
            if h > topk_height:
                bar.set_height(topk_height)
            heights.append(h)
        ax.bar_label(container, labels = [f'{h:.2f}\n↑' if h > topk_height else "" for h in heights ])


    

