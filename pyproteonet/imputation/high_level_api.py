from typing import Dict, List, Optional
from functools import partial

from pyproteonet.data import Dataset
from tqdm.auto import tqdm
import numpy as np


def impute_molecule(
    dataset: Dataset,
    molecule: str,
    column: str,
    methods: Optional[List[str]] = None,
    result_columns: Optional[List[str]] = None,
    mnar_percentile: float = 1,
    knn_k: int = 5,
):
    if isinstance(methods, str):
        methods = [methods]
    if methods is None:
        methods = [
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
            "collaborative_filtering"
        ]
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
    if 'collaborative_filtering' in methods_set:
        from pyproteonet.imputation.dnn.collaborative_filtering import collaborative_filtering_impute
        method_fns['collaborative_filtering'] = collaborative_filtering_impute
    for m, rc in tqdm(zip(methods, result_columns), total=len(methods)):
        print(m, rc)
        dataset.values[molecule][rc] = method_fns[m](
            dataset=dataset, molecule=molecule, column=column
        )