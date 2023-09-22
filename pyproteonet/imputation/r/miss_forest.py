from typing import Optional, Literal
import multiprocessing

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

import numpy as np
import pandas as pd
from ...data.dataset import Dataset

if not robjects.r('"missForest" %in% rownames(installed.packages())')[0]:
    robjects.r('install.packages("missForest", repos = "https://cloud.r-project.org")')
miss_forest = importr("missForest")
if not robjects.r('"doParallel" %in% rownames(installed.packages())')[0]:
    robjects.r('install.packages("doParallel", repos = "https://cloud.r-project.org")')
do_parallel = importr("doParallel")
if not robjects.r('"doRNG" %in% rownames(installed.packages())')[0]:
    robjects.r('install.packages("doRNG", repos = "https://cloud.r-project.org")')
do_rng = importr("doRNG")


def impute_miss_forest(
    dataset: Dataset,
    molecule: str,
    column: str,
    result_column: Optional[str] = None,
    proteins_as_variables:bool = False,
    ntree=100,
    **kwds
):
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    mat = matrix.to_numpy()
    if proteins_as_variables:
        mat = mat.T
    mask = (np.isnan(mat).sum(axis=0) < mat.shape[0] - 1)#missForest requires at least two samples
    #Enable multithreading
    # if proteins_as_variables:
    #     parallelize = 'variables'
    #     workers = min(multiprocessing.cpu_count(), ntree, mat.shape[1], 4)
    #     print(workers)
    #     do_parallel.registerDoParallel(cores=workers)
    #     do_rng.registerDoRNG(seed=1.618)
    # else:
    #     parallelize = 'no'
    with (robjects.default_converter + numpy2ri.converter).context():
        print(mat[:, mask].shape)
        imp = miss_forest.missForest(mat[:, mask], parallelize='no', ntree=ntree, **kwds)['ximp']
    mat[:, mask] = imp
    if proteins_as_variables:
        mat = mat.T
    assert mat.shape == matrix.shape

    matrix.loc[:, :] = mat
    col_means = matrix.mean()
    for c in matrix:
        col = matrix[c]
        col[col.isna()] = col_means[c]

    if result_column is not None:
        dataset.set_samples_value_matrix(molecule=molecule, column=result_column, matrix=matrix)
    
    vals = matrix.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals