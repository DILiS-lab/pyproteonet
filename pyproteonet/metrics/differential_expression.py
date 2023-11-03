from typing import Union, List
import math

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from ..data.dataset import Dataset

def find_des(dataset: Dataset, molecule:str, columns: Union[str, List[str]], group1_samples: List[str], group2_samples: List[str], is_log: bool = True):
    if isinstance(columns, str):
        columns = [columns]
    des = pd.DataFrame(index=dataset.molecules[molecule].index)
    pvalues_mat = pd.DataFrame(index=dataset.molecules[molecule].index)
    fcs = pd.DataFrame(index=dataset.molecules[molecule].index)
    for c in columns:
        mat = dataset.get_samples_value_matrix(molecule='protein_group', column=c)
        if is_log:
            mat = math.e**mat
        log_mat = np.log(mat)
        test_res = ttest_ind(log_mat[group1_samples].to_numpy(), log_mat[group2_samples].to_numpy(), axis=1, nan_policy='omit')
        pvalues = pd.Series(test_res.pvalue.data, index=mat.index)
        pvalues[pvalues.isna()] = 1.0
        correction_res = multipletests(pvals=pvalues, alpha=0.05, method='fdr_bh')
        des[c] = correction_res[0]
        pvalues_mat[c] = correction_res[1]
        fc = mat[group1_samples].median(axis=1).to_numpy() /  mat[group2_samples].median(axis=1).to_numpy()
        fcs[c] = fc
    return des, pvalues_mat, fcs