from typing import Optional, List

import matplotlib
import numpy as np
import seaborn as sns

from ..data.dataset import Dataset


def plot_clustermap(
    dataset: Dataset,
    molecule: str,
    column: str,
    logarithmize: bool = True,
    category_molecule_column: Optional[str] = None,
    category_colors: Optional[List] = None,
    row_cluster: bool = True,
    **kwargs
)-> 'ClusterGrid':
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    if logarithmize:
        mat = np.log(mat)
    mat[mat.isna()] = 0
    row_colors = None
    if category_molecule_column is not None:
        if category_colors is None:
            category_colors = matplotlib.colormaps['tab20'].colors
        row_colors = dataset.molecules[molecule][category_molecule_column]
        unique_cats = row_colors.unique()
        if len(category_colors) < len(unique_cats):
            raise KeyError(f"The provided list of category_colors only has {len(category_colors)} entries but there are {len(unique_cats)} categories.")
        lut = dict(zip(unique_cats, category_colors))
        row_colors = row_colors.map(lut)
    cluster_map = sns.clustermap(mat, row_colors=row_colors, row_cluster=row_cluster, **kwargs)
    cluster_map.ax_heatmap.tick_params(right=False)
    cluster_map.ax_heatmap.set_yticks([], [])
    cluster_map.ax_heatmap.set_ylabel(f"$Log_e$({column})")
    cluster_map.ax_heatmap.set_xlabel("Sample")
    return cluster_map