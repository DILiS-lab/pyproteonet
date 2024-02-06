from typing import Optional
from functools import lru_cache

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import numpy as np
from ...data.dataset import Dataset

@lru_cache(maxsize=1)
def _init():
    if not robjects.r('"missForest" %in% rownames(installed.packages())')[0]:
        robjects.r('install.packages("missForest", repos = "https://cloud.r-project.org")')
    miss_forest = importr("missForest")
    if not robjects.r('"doParallel" %in% rownames(installed.packages())')[0]:
        robjects.r('install.packages("doParallel", repos = "https://cloud.r-project.org")')
    do_parallel = importr("doParallel")
    if not robjects.r('"doRNG" %in% rownames(installed.packages())')[0]:
        robjects.r('install.packages("doRNG", repos = "https://cloud.r-project.org")')
    do_rng = importr("doRNG")
    return miss_forest


def miss_forest_impute(
    dataset: Dataset,
    molecule: str,
    column: str,
    result_column: Optional[str] = None,
    molecules_as_variables:bool = True,
    ntree=100,
    **kwds
):
    """Impute using the MissForest method as implemented by the missForest R package which uses a random forest for missing value prediction. 

    Args:
        dataset (Dataset): Dataset to impute.
        molecule (str): Molecule type to impute (e.g. protein, peptide etc.).
        column (str): Name of the value column to impute.
        result_column (Optional[str], optional): If given, name of the value column to store the imputed values in. Defaults to None.
        molecules_as_variables (bool, optional): Whether to transpose the input matrix before imputation (treating molecules instead of samples as variables for the random forest). Defaults to True.
        ntree (int, optional): Number of trees to use for the random forest. Defaults to 100.

    Returns:
        pd.Series: The imputed values.
    """
    miss_forest = _init()
    matrix = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    mat = matrix.to_numpy()
    if molecules_as_variables:
        mat = mat.T
    mask = (np.isnan(mat).sum(axis=0) < mat.shape[0] - 1)#missForest requires at least two samples
    # TODO: multithreading does not seem to work at the moment, investigate this further
    # if proteins_as_variables:
    #     parallelize = 'variables'
    #     workers = min(multiprocessing.cpu_count(), ntree, mat.shape[1], 4)
    #     print(workers)
    #     do_parallel.registerDoParallel(cores=workers)
    #     do_rng.registerDoRNG(seed=1.618)
    # else:
    #     parallelize = 'no'
    with (robjects.default_converter + numpy2ri.converter).context():
        imp = miss_forest.missForest(mat[:, mask], parallelize='no', ntree=ntree, **kwds)['ximp']
    mat[:, mask] = imp
    if molecules_as_variables:
        mat = mat.T
    assert mat.shape == matrix.shape

    matrix.loc[:, :] = mat
    col_means = matrix.mean()
    for c in matrix:
        col = matrix[c]
        col[col.isna()] = col_means[c]

    if result_column is not None:
        dataset.set_wf(molecule=molecule, column=result_column, matrix=matrix)
    
    vals = matrix.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals