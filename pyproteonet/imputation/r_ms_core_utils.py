from typing import Optional

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

from ..data.dataset import Dataset
from ..utils.pandas import matrix_to_multiindex

if not robjects.r('"MsCoreUtils" %in% rownames(installed.packages())')[0]:
    robjects.r("install.packages('MsCoreUtils', repos='http://cran.us.r-project.org')")
robjects.r('library("MsCoreUtils")')
ms_core_utils = importr('MsCoreUtils')


def impute_miss_forest(dataset: Dataset, molecule: str, column: str, result_column: Optional[str] = None, **kwargs):
    if not robjects.r('"missForest" %in% rownames(installed.packages())')[0]:
        robjects.r("install.packages('missForest', repos='http://cran.us.r-project.org')")
    miss_forest = importr('missForest')
    ms_core_utils = importr('MsCoreUtils')
    mat = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    #with (robjects.default_converter + pandas2ri.converter).context():
    with (robjects.default_converter + numpy2ri.converter).context():
        mat = mat.loc[~(mat.isna().all(axis=1))]
        robjects.globalenv['mat'] = mat.to_numpy()
        import pdb; pdb.set_trace()
        #res = robjects.r('impute_matrix(mat, method="RF")')
        res = ms_core_utils.impute_matrix(mat, method = 'RF', **kwargs)
        import pdb; pdb.set_trace()
    if result_column is not None:
        dataset.set_samples_value_matrix(matrix=res, molecule=molecule, column=result_column)
    return matrix_to_multiindex(res)