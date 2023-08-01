from typing import List, Union, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


def draw_normal_log_space(
    molecule_set: MoleculeSet,
    log_mu: float = 10,
    log_sigma: float = 2,
    num_samples: int = 10,
    molecule: str = "protein",
    column: str = "abundance",
    log_error_mu: float = 0,
    log_error_sigma: float = 0,
    random_seed=None,
) -> Dataset:
    """Draws molecule values from normal distribution in log space and copies result across multiple samples to create a Dataset.

    Args:
        molecule_set: MoleculeSet for which values are drawn.
        log_mu: Abundance mean in log space. Defaults to 10.
        log_sigma: Abundance std in log space. Defaults to 2.
        num_samples: Number of samples in result Dataset. Defaults to 10.
        molecule: Molecule type to draw values for. Defaults to 'protein'.
        column: Column to save drawn values in. Defaults to 'abundance'.
        log_error_mu: Mean of normal distributed error term in log space. Default to 0.
        log_error_sigma: Standard deviation of normal distributed error term in log space. Default to 0.
        random_seed: Seed for random generator. Defaults to None.

    Returns:
        Dataset: The created Dataset.
    """
    num_proteins = len(molecule_set.molecules[molecule])
    rng = np.random.default_rng(seed=random_seed)
    log_values = rng.normal(loc=log_mu, scale=log_sigma, size=num_proteins)
    log_errors = rng.normal(loc=log_error_mu, scale=log_error_sigma, size=num_proteins)
    dataset = Dataset(molecule_set=molecule_set)
    for i in range(num_samples):
        values = pd.DataFrame(data={column:np.exp(log_values + log_errors)}, index=molecule_set.molecules[molecule].index)
        values = {molecule: values}
        dataset.create_sample(name=f"sample_{i}", values=values)
    return dataset