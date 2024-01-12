"""
Imputation methods for missing values in proteomics data. Next to several imputation methods implemented by different Python packages
and newly proposed graph neural network imputation methods this package also wraps imputation methods from popular R packages.
"""

from .sklearn import knn_impute, iterative_impute
from .fancyimpute import iterative_svd_impute
from .multistep import multistep_top3_impute
from .simple import across_sample_aggregate_impute
from .simple import min_det_impute
from .high_level_api import impute_molecule
from .random_forest import random_forest_impute
from .dnn.autoencoder import auto_encoder_impute
from .dnn.collaborative_filtering import collaborative_filtering_impute