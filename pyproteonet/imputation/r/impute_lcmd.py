from typing import Optional

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

from ...data.dataset import Dataset
from ...utils.pandas import matrix_to_multiindex

if not robjects.r('"BiocManager" %in% rownames(installed.packages())')[0]:
    robjects.r('install.packages("BiocManager", repos = "https://cloud.r-project.org")')
bioc_manager = importr("BiocManager")
if not robjects.r('"imputeLCMD" %in% rownames(installed.packages())')[0]:
    bioc_manager.install("imputeLCMD", ask=False)
robjects.r("library(imputeLCMD)")
r_min_prob = robjects.r('imputeLCMD::impute.MinProb')


def impute_min_prob(dataset: Dataset, molecule: str, column: str, q:float=0.01, tune_sigma:float = 1, result_column: Optional[str] = None,**kwargs):
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    with (robjects.default_converter + pandas2ri.converter).context():
        res = r_min_prob(mat, q=q, tune_sigma=tune_sigma)
    res.index = res.index.astype(mat.index.dtype)#R transforms indices to str for some reason
    if result_column is not None:
        dataset.set_samples_value_matrix(matrix=res, molecule=molecule, column=result_column)
    return matrix_to_multiindex(res)