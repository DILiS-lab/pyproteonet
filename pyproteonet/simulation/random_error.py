from typing import List, Union

from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_numpy_random_generator
from ..data.dataset import Dataset


def add_positive_gaussian(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    mu: float = 0,
    sigma: float = 1,
    inplace: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
) -> Dataset:
    """For every sample and value of the given molecule and column add the absolute value of an error drawn from a normal distribution.
        Can be used to simulate background noise observed during measurements.

    Args:
        dataset (Dataset): Input Dataset.
        molecule (str, optional): Molecule type to apply random error to. Defaults to "protein".
        column (str, optional): Column to apply error to. Defaults to "abundance".
        result_column (str, optional): Column to write result to. Defaults to the input column if not given.
        mean (float, optional): _description_. Defaults to 0.
        std (float, optional): _description_. Defaults to 1.
        inplace (bool, optional): Whether to copy the datase before scaling. Defaults to False.
        random_seed (Optional[int], optional): Random seed used for sampling the scaling factor distribution. Defaults to None.

    Returns:
        Dataset: Result Dataset with random error applied.
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)
    for sample in dataset.samples:
        vals = sample.values[molecule].loc[:, column]
        error = rng.normal(loc=mu, scale=sigma, size=len(vals))
        sample.values[molecule][result_column] = vals + np.abs(error)
    return dataset


def multiply_exponential_gaussian(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    sigma: float = 0.33,
    inplace: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
    ids: Optional[pd.Index] = None
) -> Dataset:
    """For every sample and value of the given molecule and column multiply the value by e**error, with error drawn from a normal distribution.

    Args:
        dataset (Dataset): Input Dataset.
        molecule (str, optional): Molecule type to apply random error to. Defaults to "protein".
        column (str, optional): Column to apply error to. Defaults to "abundance".
        result_column (str, optional): Column to write result to. Defaults to the input column if not given.
        std (float, optional): Standard deviation of random normal error distribution. Defaults to 1.
        inplace (bool, optional): Whether to copy the datase before scaling. Defaults to False.
        random_seed (Optional[int], optional): Random seed used for sampling the scaling factor distribution. Defaults to None.

    Returns:
        Dataset: Result Dataset with random error applied.
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)
    for sample in dataset.samples:
        vals = sample.values[molecule].loc[:, column].copy()
        if ids is None:
            error = rng.normal(loc=0, scale=sigma, size=len(vals))
            vals = vals * np.exp(error)
        else:
            error = rng.normal(loc=0, scale=sigma, size=len(ids))
            vals.loc[ids] = vals.loc[ids] * np.exp(error)
        sample.values[molecule][result_column] = vals
    return dataset


def add_std_correlated_gaussian(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    sigma_factor: float = 0.33,
    clip: bool = True,
    inplace: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
) -> Dataset:
    """For every sample and value of the given molecule and column add the absolute value of an error
        drawn from a normal distribution whose stand deviation is the original value times the std_factor.

    Args:
        dataset (Dataset): Input Dataset.
        molecule (str, optional): Molecule type to apply random error to. Defaults to "protein".
        column (str, optional): Column to apply error to. Defaults to "abundance".
        result_column (str, optional): Column to write result to. Defaults to the input column if not given.
        std_factor (float, optional): Standard deviation of random normal error distribution is set to std_factor * input_column_value . Defaults to 1.
        clip (bool, optional): Whether to clip the values to always be >=0. Default to True.
        inplace (bool, optional): Whether to copy the datase before scaling. Defaults to False.
        random_seed (Optional[int], optional): Random seed used for sampling the scaling factor distribution. Defaults to None.

    Returns:
        Dataset: Result Dataset with random error applied.
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)

    for sample in dataset.samples:
        vals = sample.values[molecule].loc[:, column]
        error = rng.normal(loc=0, scale=sigma_factor * vals, size=len(vals))
        vals = vals + error
        if clip:
            vals = np.clip(vals, a_min=0, a_max=None)
        sample.values[molecule][result_column] = vals
    return dataset


def poisson_error(
    dataset: Dataset,
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    inplace: bool = False,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
):
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = get_numpy_random_generator(seed=random_seed)
    for sample in dataset.samples:
        vals = sample.values[molecule].loc[:, column]
        mask = ~vals.isna()
        vals = vals[mask]
        sample.values[molecule].loc[mask, result_column] = rng.poisson(lam=vals)
    return dataset
