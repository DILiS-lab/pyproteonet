from typing import Union, List
import math

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from ..data.dataset import Dataset

#TODO: add docstrings
def find_des(dataset: Dataset, molecule:str, columns: Union[str, List[str]],
             nominator_samples: List[str], denominator_samples: List[str],
             max_pvalue=0.05, min_fc=2, is_log: bool = False):
    if isinstance(columns, str):
        columns = [columns]
    des = pd.DataFrame(index=dataset.molecules[molecule].index)
    pvalues_mat = pd.DataFrame(index=dataset.molecules[molecule].index)
    fcs = pd.DataFrame(index=dataset.molecules[molecule].index)
    for c in columns:
        mat = dataset.get_samples_value_matrix(molecule=molecule, column=c)
        if is_log:
            mat = math.e**mat
        log_mat = np.log2(mat)
        test_res = ttest_ind(log_mat[nominator_samples].to_numpy(), log_mat[denominator_samples].to_numpy(), axis=1, nan_policy='omit')
        pvalues = pd.Series(test_res.pvalue.data, index=mat.index)
        pvalues[pvalues.isna()] = 1.0
        correction_res = multipletests(pvals=pvalues, alpha=max_pvalue, method='fdr_bh')
        des[c] = correction_res[0]
        pvalues_mat[c] = correction_res[1]
        fc = mat[nominator_samples].median(axis=1).to_numpy() / mat[denominator_samples].median(axis=1).to_numpy()
        des[c] = des[c] & np.abs(np.log(fc)) > np.log(min_fc)
        fcs[c] = fc
    return des, pvalues_mat, fcs


#TODO: add docstrings
def evaluate_des(dataset: Dataset, molecule: str, columns: Union[str, List[str]], numerator_samples: List[str], 
                 denominator_samples: List[str], gt_fc: pd.Series, min_fc: float = 1.5, max_pvalue: float = 0.05,
                 is_log: bool = False)->pd.DataFrame:
    des, pvalues, fc = find_des(dataset=dataset, molecule=molecule, columns=columns,
                                nominator_samples=numerator_samples, denominator_samples=denominator_samples,
                                min_fc=min_fc, max_pvalue=max_pvalue, is_log=is_log)
    gt_fc = pd.DataFrame(index=gt_fc.index, data=np.stack([gt_fc.values]*fc.shape[1], axis=1), columns=fc.columns)
    gt_de = (gt_fc > min_fc) | (gt_fc < 1/min_fc)
    correct_higher = (gt_fc > min_fc) & (fc > min_fc)
    correct_lower = (gt_fc < 1/min_fc) & (fc < min_fc)
    correctly_found = des & (correct_higher | correct_lower) & gt_de
    correctly_not_found = (~des) & (~gt_de)
    prec_rec_eval = pd.DataFrame({'Recall':correctly_found.sum() / gt_de.sum()})
    prec_rec_eval['Precision'] = correctly_found.sum() / des.sum()
    prec_rec_eval['Specificity'] = correctly_not_found.sum() / (~des).sum()
    prec_rec_eval['Accuracy'] = (gt_de == des).sum() / gt_de.count()
    prec_rec_eval['FP Rate'] = (des & (gt_de)).sum() / (~gt_de).sum()
    return prec_rec_eval