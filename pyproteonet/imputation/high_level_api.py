from typing import Dict, List, Optional
from functools import partial
import time

from pyproteonet.data import Dataset
from tqdm.auto import tqdm
import numpy as np

ALL_IMPUTATION_METHODS = [
            "minprob",
            "mindet",
            "mean",
            "bpca",
            "bpca_t",
            "missforest",
            "missforest_t",
            "knn",
            "isvd",
            "iterative",
            "dae",
            "vae",
            "cf"
        ]

def impute_molecule(
    dataset: Dataset,
    molecule: str,
    column: str,
    methods: Optional[List[str]] = None,
    result_columns: Optional[List[str]] = None,
    mnar_percentile: float = 1,
    knn_k: int = 5,
    measure_runtime: bool = True,
):
    """
    Imputes missing values in a specific molecule and column of a dataset using various imputation methods.
    Currently supported methods are:

    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Method String | Details                                                                                                                                                                    |
    +===============+============================================================================================================================================================================+
    | mindet        | MinDet imputation (see :func:`~pyproteonet.imputation.simple.min_det_impute`)                                                                                              |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | minprob       | MinProb imputation (see :func:`~pyproteonet.imputation.r.impute_lcmd.min_prob_impute`)                                                                                     |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | mean          | Mean imputation across samples (see :func:`~pyproteonet.imputation.simple.across_sample_aggregate_impute` with `method` argument set to "mean")                            |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | bpca          | BPCA imputation (see :func:`~pyproteonet.imputation.r.pca_methods` with `method` parameter set to "bpca")                                                                  |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | bpca_t        | BPCA imputation on transpose data (see :func:`~pyproteonet.imputation.r.pca_methods` with `method` argument set to "bpca" and `molecules_as_variables`  set to `True`)     |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | missforest    | MissForest imputation (see :func:`~pyproteonet.imputation.random_forest.random_forest_impute`)                                                                             |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | missforest_t  | MissForest imputation on transpose data (see :func:`~pyproteonet.imputation.r.miss_forest.miss_forest_impute` with  `molecule_as_variables` argument set to `True`)        |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | knn           | KNN imputation (see :func:`~pyproteonet.imputation.sklearn.knn_impute`)                                                                                                    |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | isvd          | ISVD imputation (see :func:`~pyproteonet.imputation.fancyimpute.iterative_svd_impute`)                                                                                     |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | iterative     | Iterative imputation from scikit-learn (see :func:`~pyproteonet.imputation.sklearn.iterative_impute`)                                                                      |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | dae           | Denoising Autoencoder imputation (see :func:`~pyproteonet.imputation.dnn.autoencoder.auto_encoder_impute` with `model_type` argument set to "DAE")                         |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | vae           | Variational Autoencoder imputation (see  :func:`~pyproteonet.imputation.dnn.autoencoder.auto_encoder_impute` with  `model_type` argument set to "VAE")                     |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | cf            | Collaborative Filtering imputation (see :func:`~pyproteonet.imputation.dnn.collaborative_filtering.collaborative_filtering_impute`)                                        |
    +---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

    Args:
        dataset (Dataset): The dataset containing the values to be imputed.
        molecule (str): The molecule for which missing values will be imputed.
        column (str): The column in the dataset corresponding to the molecule.
        methods (Optional[List[str]], optional): List of imputation methods to be used. Defaults to None.
        result_columns (Optional[List[str]], optional): List of column names to store the imputed values. Defaults to None.
        mnar_percentile (float, optional): Percentile value for missing not at random (MNAR) imputation methods. Defaults to 1.
        knn_k (int, optional): Number of nearest neighbors to consider for k-nearest neighbors (KNN) imputation. Defaults to 5.

    Raises:
        ValueError: If the number of methods and result columns is not the same.

    Returns:
        None
    """
    runtimes = dict()
    if isinstance(methods, str):
        methods = [methods]
    if methods is None:
        methods = ALL_IMPUTATION_METHODS
    if isinstance(result_columns, str):
        result_columns = [result_columns]
    if result_columns is None:
        result_columns = methods
    methods_set = set(methods)
    if len(methods) != len(result_columns):
        raise ValueError("Number of methods and result columns should be the same")
    method_fns = dict()
    if 'mindet' in methods_set:
        from pyproteonet.imputation.simple import min_det_impute
        method_fns['mindet'] = partial(min_det_impute, percentile=mnar_percentile)
    if 'minprob' in methods_set:
        from pyproteonet.imputation.r.impute_lcmd import min_prob_impute
        method_fns['minprob'] = partial(min_prob_impute, q=mnar_percentile / 100)
    if 'mean' in methods_set:
        from pyproteonet.imputation.simple import across_sample_aggregate_impute
        method_fns['mean'] = partial(across_sample_aggregate_impute, method="mean", all_missing_percentile=mnar_percentile)
    if 'bpca' in methods_set:
        from pyproteonet.imputation.r.pca_methods import impute_pca_method
        method_fns['bpca'] = partial(impute_pca_method, method="bpca")
    if 'bpca_t' in methods_set:
        from pyproteonet.imputation.r.pca_methods import impute_pca_method
        method_fns['bpca_t'] = partial(impute_pca_method, method="bpca", molecules_as_variables=True)
    if 'ppca' in methods_set:
        from pyproteonet.imputation.r.pca_methods import impute_pca_method
        method_fns['ppca'] = partial(impute_pca_method, method="ppca")
    if 'missforest' in methods_set:
        #We use a Python implementation of missForest, because it is faster than the R implementation for this senario
        from pyproteonet.imputation.random_forest import random_forest_impute
        method_fns['missforest'] = partial(random_forest_impute, molecules_as_variables=False)
    if 'missforest_t' in methods_set:
        from pyproteonet.imputation.r.miss_forest import miss_forest_impute
        method_fns['missforest_t'] = partial(miss_forest_impute, molecules_as_variables=True)
    if 'knn' in methods_set:
        from pyproteonet.imputation.sklearn import knn_impute
        method_fns['knn'] = partial(knn_impute, n_neighbors=knn_k)
    if 'isvd' in methods_set:
        from pyproteonet.imputation.fancyimpute import iterative_svd_impute
        method_fns['isvd'] = partial(iterative_svd_impute, rank=0.2)
    if 'iterative' in methods_set:
        from pyproteonet.imputation.sklearn import iterative_impute
        method_fns['iterative'] = iterative_impute
    if 'dae' in methods_set:
        from pyproteonet.imputation.dnn.autoencoder import auto_encoder_impute
        method_fns['dae'] = partial(auto_encoder_impute, validation_fraction=0.1, model_type='DAE')
    if 'vae' in methods_set:
        from pyproteonet.imputation.dnn.autoencoder import auto_encoder_impute
        method_fns['vae'] = partial(auto_encoder_impute, validation_fraction=0.1, model_type='VAE')
    if 'cf' in methods_set:
        from pyproteonet.imputation.dnn.collaborative_filtering import collaborative_filtering_impute
        method_fns['cf'] = collaborative_filtering_impute
    for m, rc in tqdm(zip(methods, result_columns), total=len(methods)):
        print(f'Imputing with method {m}, storing results in value column {rc}')
        runtime = time.time()
        imp = method_fns[m](
            dataset=dataset, molecule=molecule, column=column
        )
        runtime = time.time() - runtime
        runtimes[m] = runtime
        dataset.values[molecule][rc] = imp
    if measure_runtime:
        return runtimes