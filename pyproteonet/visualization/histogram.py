from typing import Union

from matplotlib import pyplot as plt
import seaborn as sbn
import numpy as np

from ..data.dataset import Dataset
from ..data.dataset_sample import DatasetSample

def plot_hist(data: Union[Dataset, DatasetSample], molecule: str, column: str = 'abundance', bins="auto", log_space: bool = False,
              ax=None, **kwds):
    if ax is None:
        fig, ax = plt.subplots()
    values, mask = data.all_values(
        molecule=molecule, return_missing_mask=True, column=column
    )
    missing_percent = mask.sum() / values.shape[0] * 100
    values = values[~mask]
    if log_space:
        values = np.log(values)
    sbn.histplot(values, ax=ax, bins=bins, **kwds)
    ax.set_title(f"{molecule} {column} ({round(missing_percent, 1)}% missing)")