from typing import Union, Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn

from ..metrics.differential_expression import find_des
from ..data.dataset import Dataset


def plot_des_volcano(
    dataset: Dataset,
    molecule: str,
    columns: Union[str, List[str]],
    numerator_samples: List[str],
    denominator_samples: List[str],
    is_log: bool = False,
    alpha: float = 0.05,
    categories: Optional[pd.Series] = None,
    ids: Optional[pd.Index] = None,
    axs: Optional[List[plt.Axes]] = None,
    s=4,
):
    if isinstance(columns, str):
        columns = [columns]
    if axs is None:
        fig, axs = plt.subplots(1, len(columns), figsize=(15, 5))
    des, p_values, fc = find_des(
        dataset=dataset,
        molecule=molecule,
        columns=columns,
        nominator_samples=numerator_samples,
        denominator_samples=denominator_samples,
        is_log = is_log
    )
    if ids is not None:
        des, p_values, fc = des.loc[ids,:], p_values.loc[ids,:], fc.loc[ids,:]
    for c, ax in zip(columns, axs):
        #fold_change = mat[numerator_samples].median(axis=1) / mat[denominator_samples].median(axis=1)
        #fold_changes[c] = fold_change
        if categories is None:
            sbn.scatterplot(x=fc[c], y=-np.log10(p_values[c]), ax=ax, s=s, linewidth=0)
        else:
            sbn.scatterplot(x=np.log2(fc[c]), y=-np.log10(p_values[c]), hue=categories, ax=ax, s=s, linewidth=0)
        ax.axhline(y=-np.log10(alpha), linestyle='dotted')
        ax.set_title(c)
        ax.set_ylabel('-$Log_{10}$(p_value)')
        ax.set_xlabel('$Log_2$(fold_change)')
        ax.legend()
    for ax in axs[len(columns):]:
        ax.set_axis_off()
    return des, p_values, fc
