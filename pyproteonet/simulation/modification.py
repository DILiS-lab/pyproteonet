from typing import Union, Optional

import numpy as np
import scipy

from .utils import get_numpy_random_generator
from ..data.dataset import Dataset


def per_molecule_random_scaling(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    beta_distr_alpha: float = 5,
    beta_distr_beta: float = 2.5,
    inplace: bool = False,
    random_seed: Optional[Union[np.random.Generator, int]] = None
) -> Dataset:
    """Draws a random factor within [0,1] for each molecule of the given type and multiplies all column values across all samples with it.

    Args:
        dataset (Dataset): Input Dataset.
        molecule (str, optional): Molecule type whose values are scaled. Defaults to "protein".
        column (str, optional): Column column to scale. Defaults to "abundance".
        result_column (str, optional): Column to write scaled value to. Defaults to the input column.
        beta_distr_alpha (float, optional): Alpha parameter of Beta distribution used to sample scaling factors. Defaults to 5.
        beta_distr_beta (float, optional): Beta parameter of Beta distribution used to sample scaling factors. Defaults to 2.5.
        inplace (bool, optional): Whether to copy the datase before scaling. Defaults to False.
        random_seed (Optional[int], optional): Random seed used for sampling the scaling factor distribution. Defaults to None.

    Returns:
        Dataset: Result Dataset with randomly scaled values
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)
    scaling_factors = scipy.stats.beta.rvs(
            a=beta_distr_alpha, b=beta_distr_beta, size=len(dataset.molecules[molecule]), random_state=rng
        )
    index = dataset.molecules[molecule].index
    for sample in dataset.samples:
        vals = sample.values[molecule].loc[index, column]
        sample.values[molecule][result_column] =  vals * scaling_factors
    return dataset

