from typing import Optional, Literal

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

import pandas as pd
from ...data.dataset import Dataset

if not robjects.r('"BiocManager" %in% rownames(installed.packages())')[0]:
    robjects.r('install.packages("BiocManager", repos = "https://cloud.r-project.org")')
bioc_manager = importr("BiocManager")
if not robjects.r('"pcaMethods" %in% rownames(installed.packages())')[0]:
    bioc_manager.install("pcaMethods", ask=False)
pca_methods = importr("pcaMethods")
ms_core_utils = importr("MsCoreUtils")

nan_to_na = robjects.r(
    """nan_to_na <- function(mat){
    replace(mat, is.na(mat), NA)
}"""
)


def impute_pca_method(
    dataset: Dataset,
    molecule: str,
    column: str,
    method: Literal['svdPca', 'ppca', 'bpca', 'svdImpute'] = "bpca",
    n_pcs: Optional[int] = None,
    result_column: Optional[str] = None,
):
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    if n_pcs is None:
        n_pcs = mat.shape[1] - 1
    with (robjects.default_converter + pandas2ri.converter).context():
        res = pca_methods.pca(nan_to_na(mat), method=method, nPcs=n_pcs, verbose=False)
        res = pca_methods.completeObs(res)
    assert res.shape == mat.shape
    matrix_imputed = pd.DataFrame(res, columns=mat.columns, index=mat.index)
    if result_column is not None:
        dataset.set_samples_value_matrix(matrix=matrix_imputed, molecule=molecule, column=result_column)
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals


def impute_local_least_squares(
    dataset: Dataset,
    molecule: str,
    column: str,
    k: int = 10,
    correlation: Literal['pearson', 'kendall', 'spearman'] = "pearson",
    all_variables: bool = True,
    maxSteps = 100,
    result_column: Optional[str] = None,
):
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    input = mat.to_numpy().T
    with (robjects.default_converter + numpy2ri.converter).context():
        res = pca_methods.llsImpute(nan_to_na(input), k=k, correlation=correlation, allVariables=all_variables, maxSteps=maxSteps, verbose=False)
        res = pca_methods.completeObs(res)
    res = res.T
    assert mat.shape == res.shape
    matrix_imputed = pd.DataFrame(res, columns=mat.columns, index=mat.index)
    if result_column is not None:
        dataset.set_samples_value_matrix(matrix=matrix_imputed, molecule=molecule, column=result_column)
    vals = matrix_imputed.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals
