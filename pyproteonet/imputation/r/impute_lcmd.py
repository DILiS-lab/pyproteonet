from typing import Optional
from functools import lru_cache

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from ...data.dataset import Dataset
from ...utils.pandas import matrix_to_multiindex

@lru_cache(maxsize=1)
def _init():
    if not robjects.r('"BiocManager" %in% rownames(installed.packages())')[0]:
        robjects.r('install.packages("BiocManager", repos = "https://cloud.r-project.org")')
    bioc_manager = importr("BiocManager")
    if not robjects.r('"imputeLCMD" %in% rownames(installed.packages())')[0]:
        bioc_manager.install("imputeLCMD", ask=False)
    robjects.r("library(imputeLCMD)")
    return robjects.r('imputeLCMD::impute.MinProb')


def min_prob_impute(dataset: Dataset, molecule: str, column: str, q:float=0.01, tune_sigma:float = 1, result_column: Optional[str] = None,**kwargs):
    """Impute using the minProb method as implemented by the imputeLCMD package which replaces missing values with a value drawn from a distribution of low values. 

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
        column (str): Name of the value column to impute.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.

    Returns:
        pd.Series: The imputed values.
    """
    r_min_prob = _init()
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    with (robjects.default_converter + pandas2ri.converter).context():
        res = r_min_prob(mat, q=q, tune_sigma=tune_sigma)
    res.index = res.index.astype(mat.index.dtype)#R transforms indices to str for some reason
    if result_column is not None:
        dataset.set_wf(matrix=res, molecule=molecule, column=result_column)
    return matrix_to_multiindex(res)