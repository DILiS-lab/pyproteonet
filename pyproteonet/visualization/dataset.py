from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from ..data.dataset import Dataset


def plot_dataset_values_violins(dataset: Dataset, molecule: str, column: str, ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots()
    pep_vals = dataset.values[molecule][column]
    pep_vals = pd.DataFrame({'abundance':np.log(pep_vals), 'Sample':pep_vals.index.get_level_values('sample')})
    overall = pep_vals.copy()
    overall['Sample'] = 'Overall'
    sns.violinplot(data=pd.concat([pep_vals, overall]), y='abundance', x='Sample', ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel(f'$Log_e({molecule} Abundance)$')