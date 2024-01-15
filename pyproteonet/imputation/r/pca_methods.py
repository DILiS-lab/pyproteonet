from typing import Optional, Literal
from functools import lru_cache

import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import pandas as pd
from ...data.dataset import Dataset

@lru_cache(maxsize=1)
def _init():
    if not robjects.r('"BiocManager" %in% rownames(installed.packages())')[0]:
        robjects.r('install.packages("BiocManager", repos = "https://cloud.r-project.org")')
    bioc_manager = importr("BiocManager")
    if not robjects.r('"MsCoreUtils" %in% rownames(installed.packages())')[0]:
        bioc_manager.install("MsCoreUtils", ask=False)
    if not robjects.r('"pcaMethods" %in% rownames(installed.packages())')[0]:
        bioc_manager.install("pcaMethods", ask=False)
    pca_methods = importr("pcaMethods")
    ms_core_utils = importr("MsCoreUtils")
    base = importr("base")
    r_mat_mul = robjects.r['%*%']

    nan_to_na = robjects.r(
    """function(mat){
        replace(mat, is.na(mat), NA)
    }"""
    )
    return pca_methods, base, r_mat_mul, nan_to_na


def impute_pca_method(
    dataset: Dataset,
    molecule: str,
    column: str,
    method: Literal['svdPca', 'ppca', 'bpca', 'svdImpute'] = "bpca",
    n_pcs: Optional[int] = None,
    result_column: Optional[str] = None,
    molecules_as_variables: bool = False,
    only_transform_missing: bool = True
):
    """Apply any principal component analysis (PCA) related imputation function as implmented by the pcaMethods R package.
       Available PCA imputation methods: svdPca, ppca, bpca, as well as svdImpute which is not a PCA method but a simple imputation method based on singular value decomposition.
       See https://bioconductor.org/packages/release/bioc/html/pcaMethods.html for more details.

        Args:
            dataset (Dataset): Dataset to impute.
            molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
            column (str): Name of the value column to impute.
            method (Literal['svdPca', 'ppca', 'bpca', 'svdImpute'], optional): Imputation method to use. Defaults to "bpca".
            n_pcs (Optional[int], optional): Number of principal components to use. If not given set to number_samples-1. Defaults to None.
            result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.
            molecules_as_variables (bool, optional): Whether to transpose the input matrix before imputation (treating molecules instead of samples as variables for the PCA). Defaults to False.
            only_transform_missing (bool, optional): Whether to only predict missing values. Otherwise all values are replaced with their PCA reconstruction. Defaults to True.

        Returns:
            pd.Series: The imputed values.
    """
    pca_methods, base, r_mat_mul, nan_to_na = _init()
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    if n_pcs is None:
        n_pcs = mat.shape[1] - 1
    mat_np = mat.to_numpy()
    mask = ~np.isnan(mat_np).all(axis=1)
    with (robjects.default_converter + numpy2ri.converter).context():
        if molecules_as_variables:
            in_ = nan_to_na(mat_np[mask, :].T)
        else:
            in_ = nan_to_na(mat_np[mask, :])
        res = pca_methods.pca(in_, method=method, nPcs=n_pcs, verbose=False)
        if only_transform_missing:
            res = pca_methods.completeObs(res)
        else:
            res = pca_methods.prep(r_mat_mul(res.slots['scores'], base.t(res.slots['loadings'])),
                                   center = res.slots['center'], scale = res.slots['scale'], reverse = True)
        if molecules_as_variables:
            res = res.T
    mat_np[mask, :] = res
    mat_np[~mask, :] = np.nanmean(mat_np, axis=0)[np.newaxis, :]
    assert mat_np.shape == mat.shape
    matrix_imputed = pd.DataFrame(mat_np, columns=mat.columns, index=mat.index)
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
    """Apply local least squares imputation as implemented by the pcaMethods R package.
       See https://rdrr.io/bioc/pcaMethods/man/llsImpute.html for more details.

        Args:
            dataset (Dataset): Dataset to impute.
            molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
            column (str): Name of the value column to impute.
            k (int, optional): Number of neighbors to use for imputation. Defaults to 10.
            correlation (Literal['pearson', 'kendall', 'spearman'], optional): Correlation measure to use for neighbor search. Defaults to "pearson".
            all_variables (bool, optional): Whether to use all variables for imputation or only the k nearest neighbors. Defaults to True.
            maxSteps ([type], optional): Maximum number of iterations. Defaults to 100.
            result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.
            
        Returns:
            pd.Series: The imputed values.
    """
    pca_methods, base, r_mat_mul, nan_to_na = _init()
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
