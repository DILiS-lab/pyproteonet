from typing import List, Union, Literal, Callable, Optional, Tuple

from scipy.stats import wilcoxon
import pandas as pd
import numpy as np
import matplotlib
from tqdm.auto import tqdm

from ..data.dataset import Dataset


def _get_metric(metric: Union[str, Callable]):
    if not isinstance(metric, str):
        return metric
    metric = metric.lower()
    if metric == "ae":
        res = lambda val, gt: ((val - gt)).abs()
    elif metric == "se":
        res = lambda val, gt: (val - gt) ** 2
    else:
        raise AttributeError(f"Metric {metric} not found!")
    return res


def paired_wilcoxon_matrix(
    dataset: Dataset,
    molecule: str,
    eval_columns: List[str],
    gt_column: str,
    comparison_metric: Union[
        Literal["AE", "SE"], Callable[[pd.Series, pd.Series], pd.Series]
    ] = "AE",
    ids: Optional[pd.Index] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    significance_level: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_less = np.full((len(eval_columns), len(eval_columns)), None)
    results_greater = np.full((len(eval_columns), len(eval_columns)), None)
    df = dataset.values[molecule].df
    if ids is not None:
        df = df.loc[ids]
    ae = pd.DataFrame(index=df.index)
    for c in eval_columns:
        ae[c] = _get_metric(comparison_metric)(df[c], df[gt_column])
    with tqdm(total=len(eval_columns) ** 2 - len(eval_columns)) as pbar:
        for c1 in range(len(eval_columns)):
            for c2 in range(len(eval_columns)):
                if c1 == c2:
                    continue
                x = ae[eval_columns[c1]]
                y = ae[eval_columns[c2]]
                mask = (~x.isna()) & (~y.isna()) & (x != 0) & (y != 0)
                x, y = x[mask], y[mask]
                res_less = wilcoxon(x, y.loc[x.index], alternative="less")
                res_greater = wilcoxon(x, y.loc[x.index], alternative="greater")
                results_less[c1, c2] = res_less
                results_greater[c1, c2] = res_greater
                pbar.update(1)
    results_less = pd.DataFrame(results_less, index=eval_columns, columns=eval_columns)
    results_greater = pd.DataFrame(results_greater, index=eval_columns, columns=eval_columns)

    vec_fun = np.vectorize(lambda x: x.pvalue if x is not None else 0.5)
    pvalues_less = vec_fun(results_less)
    pvalues_greater = vec_fun(results_greater)

    res_mat = np.zeros_like(pvalues_less, dtype=int)
    sl = significance_level
    sl = sl / (len(eval_columns) ** 2 - len(eval_columns))
    res_mat[pvalues_less < 0.05] = -1
    res_mat[pvalues_greater < 0.05] = 1
    if ax is not None:
        ax.matshow(
            res_mat,
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                "", ["green", "grey", "red"]
            ),
        )
        ax.set_xticks(range(len(eval_columns)))
        ax.set_xticklabels(eval_columns, rotation=90)
        ax.set_yticks(range(len(eval_columns)))
        ax.set_yticklabels(eval_columns, rotation=0)
        ax.grid(False)
    return results_less, results_greater
