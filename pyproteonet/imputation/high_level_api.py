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
        ]
    if isinstance(result_columns, str):
        result_columns = [result_columns]
    if result_columns is None:
        result_columns = methods
    methods = set(methods)
    if len(methods) != len(result_columns):
        raise ValueError("Number of methods and result columns should be the same")
    method_fns = dict()
    if 'mindet' in methods:
        from pyproteonet.imputation.simple import min_det_impute
        method_fns['mindet'] = partial(min_det_impute, percentile=mnar_percentile)
    if 'minprob' in methods:
        from pyproteonet.imputation.r.impute_lcmd import impute_min_prob
        method_fns['minprob'] = partial(impute_min_prob, q=mnar_percentile / 100)
    if 'mean' in methods:
        from pyproteonet.imputation.simple import across_sample_aggregate
        method_fns['mean'] = partial(across_sample_aggregate, method="mean", all_missing_percentile=mnar_percentile)
    if 'bpca' in methods:
        from pyproteonet.imputation.r.pca_methods import impute_pca_method
        method_fns['bpca'] = partial(impute_pca_method, method="bpca")
    if 'bpca_t' in methods:
        method_fns['bpca_t'] = partial(impute_pca_method, method="bpca", molecules_as_variables=True)
    if 'ppca' in methods:
        from pyproteonet.imputation.r.pca_methods import impute_pca_method
        method_fns['ppca'] = partial(impute_pca_method, method="ppca")
    if 'missforest' in methods:
        #We use a Python implementation of missForest, because it is faster than the R implementation for this senario
        from pyproteonet.imputation.random_forrest import missing_forrest_impute
        method_fns['missforest'] = partial(missing_forrest_impute, molecules_as_variables=False)
    if 'missforest_t' in methods:
        from pyproteonet.imputation.r.miss_forest import impute_miss_forest
        method_fns['missforest_t'] = partial(impute_miss_forest, molecules_as_variables=True)
    if 'knn' in methods:
        from pyproteonet.imputation.sklearn import knn_impute
        method_fns['knn'] = partial(knn_impute, n_neighbors=knn_k)
    if 'isvd' in methods:
        from pyproteonet.imputation.fancyimpute import iterative_svd_impute
        method_fns['isvd'] = partial(iterative_svd_impute, rank=0.2)
    if 'iterative' in methods:
        from pyproteonet.imputation.sklearn import iterative_impute
        method_fns['iterative'] = iterative_impute
    if 'dae' in methods:
        from pyproteonet.imputation.dnn.autoencoder import impute_auto_encoder
        method_fns['dae'] = partial(impute_auto_encoder, validation_fraction=0.1, model_type='DAE')
    if 'vae' in methods:
        from pyproteonet.imputation.dnn.autoencoder import impute_auto_encoder
        method_fns['vae'] = partial(impute_auto_encoder, validation_fraction=0.1, model_type='VAE')
    for m, rc in tqdm(zip(methods, result_columns), total=len(methods)):
        dataset.values[molecule][rc] = method_fns[m](
            dataset=dataset, molecule=molecule, column=column
        )