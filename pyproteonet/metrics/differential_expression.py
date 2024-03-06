from typing import Union, List, Tuple, Optional
import math

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from ..data.dataset import Dataset

def find_des(dataset: Dataset, molecule:str, columns: Union[str, List[str]],
             nominator_samples: List[str], denominator_samples: List[str],
             max_pvalue=0.05, min_fc=2, is_log: bool = False, assume_equal_var: bool = True,
             ids: Optional[pd.Index] = None)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Find differentially expressed molecules usign a t-test and multitest correction (benjamin hochberg).

    Args:
        dataset (Dataset): The dataset to find differentially expressed molecules in.
        molecule (str): The molecule type to find differentially expressed molecules for.
        columns (Union[str, List[str]]): The value column(s) containing the abundance values of potentially differentially expressed molecules.
        nominator_samples (List[str]): List of samples names to use as nominator when computing fold change.
        denominator_samples (List[str]): List of samples names to use as denominator when computing fold change.
        max_pvalue (float, optional): P value to as significance threshold . Defaults to 0.05.
        min_fc (int, optional): Minimum fold change required to be considered as differentially expressed. Works for both increase and decrease in abundance (e.g. a min. fold change of 2 results in both a fold change of 2 and 0.5 being considered as potentially differentially expressed.). Defaults to 2.
        is_log (bool, optional): Whether the column values are logarithmized. Defaults to False.
        ids (Optional[pd.Index], optional): Ids of molecules to consider. Defaults to None (all molecules of the given molecule type are considered).

    Returns:
        Tuple(pd.DataFrame, pd.DataFrame, pd.DataFrame): Tuple of three dataframes representing differential expressions, p-values and fold changes.
    """
    if isinstance(columns, str):
        columns = [columns]
    if ids is not None:
        ids = ids.intersection(dataset.molecules[molecule].index)
    des = pd.DataFrame(index=dataset.molecules[molecule].index if ids is None else ids)
    pvalues_mat = pd.DataFrame(index=dataset.molecules[molecule].index if ids is None else ids)
    fcs = pd.DataFrame(index=dataset.molecules[molecule].index if ids is None else ids)
    for c in columns:
        mat = dataset.get_samples_value_matrix(molecule=molecule, column=c)
        if ids is not None:
            mat = mat.loc[ids, :]
        if is_log:
            mat = math.e**mat
        log_mat = np.log2(mat)
        eps = np.finfo(np.finfo(float).eps).eps
        mask = (log_mat[nominator_samples].var(axis=1) > eps) & (log_mat[denominator_samples].var(axis=1) > eps)
        test_mat = log_mat.loc[mask, :]
        test_res = ttest_ind(test_mat[nominator_samples].to_numpy(), test_mat[denominator_samples].to_numpy(), axis=1, nan_policy='omit',
                             equal_var=assume_equal_var)
        pvalues = pd.Series(1.0, index=mat.index)
        pvalues.loc[mask] = test_res.pvalue
        correction_res = multipletests(pvals=pvalues, alpha=max_pvalue, method='fdr_bh')
        des[c] = correction_res[0]
        pvalues_mat[c] = correction_res[1]
        fc = mat[nominator_samples].median(axis=1).to_numpy() / mat[denominator_samples].median(axis=1).to_numpy()
        des[c] = des[c] & np.abs(np.log(fc)) > np.log(min_fc)
        fcs[c] = fc
    return des, pvalues_mat, fcs


def evaluate_des(dataset: Dataset, molecule: str, columns: Union[str, List[str]], numerator_samples: List[str], 
                 denominator_samples: List[str], gt_fc: pd.Series, min_fc: float = 1.5, max_pvalue: float = 0.05,
                 is_log: bool = False, absolute_metrics: bool = False, ids: Optional[pd.Index] = None)->pd.DataFrame:
    """Compares the results of finding differentially expressed molecule to known ground troth differential expressiosn according to a ground truth fold change.
       Evaluation is done with respect to Precision, Recall, Specificity, Accuracy, FP Rate and F1 Score (or corresponding absolute metrics if absolute_metrics is True).

    Args:
        dataset (Dataset): The dataset to find differentially expressed molecules in.
        molecule (str): The molecule type to find differentially expressed molecules for.
        columns (Union[str, List[str]]): The value column(s) containing the abundance values of potentially differentially expressed molecules.
        nominator_samples (List[str]): List of samples names to use as nominator when computing fold change.
        denominator_samples (List[str]): List of samples names to use as denominator when computing fold change.
        gt_fc (pd.Series): Ground truth fold change for each molecule.
        min_fc (int, optional): Minimum fold change required to be considered as differentially expressed. Works for both increase and decrease in abundance (e.g. a min. fold change of 2 results in both a fold change of 2 and 0.5 being considered as potentially differentially expressed.). Defaults to 2.
        max_pvalue (float, optional): P value to as significance threshold . Defaults to 0.05.
        is_log (bool, optional): Whether the column values are logarithmized. Defaults to False.
        absolute_metrics (bool, optional): Whether to return absolute numbers of correctly/incorrectly found DEs or whether to use relative metrics (Precision, Recall ...). Defaults to False.
        ids (Optional[pd.Index], optional): Ids of molecules to consider. Defaults to None (all molecules of the given molecule type are considered).

    Returns:
        pd.DataFrame: The evaluation results according to the calculated metrics.
    """
    des, pvalues, fc = find_des(dataset=dataset, molecule=molecule, columns=columns,
                                nominator_samples=numerator_samples, denominator_samples=denominator_samples,
                                min_fc=min_fc, max_pvalue=max_pvalue, is_log=is_log, ids=ids)
    if ids is not None:
        gt_fc = gt_fc.loc[ids]
    gt_fc = pd.DataFrame(index=gt_fc.index, data=np.stack([gt_fc.values]*fc.shape[1], axis=1), columns=fc.columns)
    gt_de = (gt_fc > min_fc) | (gt_fc < 1/min_fc)
    correct_higher = (gt_fc > min_fc) & (fc > min_fc)
    correct_lower = (gt_fc < 1/min_fc) & (fc < min_fc)
    correctly_found = des & (correct_higher | correct_lower) & gt_de
    correctly_not_found = (~des) & (~gt_de)
    if absolute_metrics:
        prec_rec_eval = pd.DataFrame({'Correct DE':correctly_found.sum()})
        prec_rec_eval['Correct no DE'] = correctly_not_found.sum() 
        prec_rec_eval['Correctly Classified'] = (gt_de == des).sum() 
        prec_rec_eval['False Positives'] = (des & (~gt_de)).sum()
    else:
        prec_rec_eval = pd.DataFrame({'Recall':correctly_found.sum() / gt_de.sum()})
        prec_rec_eval['Precision'] = correctly_found.sum() / des.sum()
        prec_rec_eval['Specificity'] = correctly_not_found.sum() / (~gt_de).sum()
        prec_rec_eval['Accuracy'] = (gt_de == des).sum() / gt_de.count()
        prec_rec_eval['FP Rate'] = (des & (~gt_de)).sum() / (~gt_de).sum()
        prec_rec_eval['F1 Score'] = 2 * prec_rec_eval['Precision'] * prec_rec_eval['Recall'] / (prec_rec_eval['Precision'] + prec_rec_eval['Recall'])
    return prec_rec_eval