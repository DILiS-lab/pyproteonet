from typing import List, Union, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

from .utils import get_numpy_random_generator
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


def draw_normal_log_space(
    molecule_set: MoleculeSet,
    log_mu: float = 10,
    log_sigma: float = 2,
    samples: Union[int, List[str]] = 10,
    molecule: str = "protein",
    column: str = "abundance",
    log_error_mu: float = 0,
    log_error_sigma: float = 0,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
) -> Dataset:
    """Draws molecule values from normal distribution in log space and copies result across multiple samples to create a Dataset.

    Args:
        molecule_set: MoleculeSet for which values are drawn.
        log_mu: Abundance mean in log space. Defaults to 10.
        log_sigma: Abundance std in log space. Defaults to 2.
        samples: Number of samples as int or list of sample names for the resulting Dataset. Defaults to 10.
        molecule: Molecule type to draw values for. Defaults to 'protein'.
        column: Column to save drawn values in. Defaults to 'abundance'.
        log_error_mu: Mean of normal distributed error term in log space. Default to 0.
        log_error_sigma: Standard deviation of normal distributed error term in log space. Default to 0.
        random_seed: Seed for random generator. Defaults to None.

    Returns:
        Dataset: The created Dataset.
    """
    num_proteins = len(molecule_set.molecules[molecule])
    rng = get_numpy_random_generator(seed=random_seed)
    log_values = rng.normal(loc=log_mu, scale=log_sigma, size=num_proteins)
    dataset = Dataset(molecule_set=molecule_set)
    if isinstance(samples, int):
        samples = [f"sample{i}" for i in range(samples)]
    for sample in samples:
        log_errors = rng.normal(loc=log_error_mu, scale=log_error_sigma, size=num_proteins)
        values = pd.DataFrame(data={column:np.exp(log_values + log_errors)}, index=molecule_set.molecules[molecule].index)
        values = {molecule: values}
        dataset.create_sample(name=sample, values=values)
    return dataset