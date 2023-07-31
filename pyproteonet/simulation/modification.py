from typing import List, Union, Optional

import numpy as np
import scipy

from ..data.dataset import Dataset


def per_molecule_random_scaling(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    beta_distr_alpha: float = 5,
    beta_distr_beta: float = 2.5,
    inplace: bool = False,
    random_seed: Optional[int] = None
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
    rng = np.random.default_rng(seed=random_seed)
    scaling_factors = scipy.stats.beta.rvs(
            a=beta_distr_alpha, b=beta_distr_beta, size=len(dataset.molecules[molecule]), random_state=rng
        )
    index = dataset.molecules[molecule].index
    for sample in dataset.samples:
        vals = sample.values[molecule].loc[index, column]
        sample.values[molecule][result_column] =  vals * scaling_factors
    return dataset

def introduce_random_condition(
    dataset: Dataset,
    amount_affected: Union[float, int] = 0.15,
    log2_cond_factor_mean: float = 0,
    log2_cond_factor_std: float = 1,
    samples_affected: Union[List[str], int, float] = 0.5,
    molecule: str = "protein",
    column: str = "abundance",
    result_column: Optional[str] = None,
    inplace: bool = False,
    random_seed: Optional[int] = None
) -> Dataset:
    """Samples condition factors for a subset of randomly chosen molecues
       and multiplies the molecules values of affected with those factors.

    Args:
        dataset (Dataset): Input Dataset.
        amount_affected (Union[float, int], optional): Number of affected molecules.
            Floats in [0, 1.0] are intepreted as relative fractions. Defaults to 0.15.
        log2_cond_factor_mean (float, optional): Mean of normal distribution in log2 space,
            used for sampling conditin factors. Defaults to 0.
        log2_cond_factor_std (float, optional): Std of normal distribution in log2 space,
            used for sampling conditin factors. Defaults to 1.
        samples_affected (Union[List[str], int, float], optional): If list, will be interpreted as names of samples,
            if float will be interpreted as fraction of samples, if int will be interpreted as number of fractions.
            Defaults to 0.5.
        molecule (str, optional): Molecule type to draw values for. Defaults to 'protein'.
        column (str, optional): Column to apply condition to Defaults to 'abundance'.
        result_column (str, optional): Column to write result to. Defaults to the input column if not given.
        inplace (float, optional): Whether to copy the dataset before introducing the condition.
        random_seed (_type_, optional): Seed for random generator. Defaults to None.


    Returns:
        Dataset:  The resulting Dataset with condition applied.
    """
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    rng = np.random.default_rng(seed=random_seed)
    condition_samples = dataset.sample_names
    if isinstance(samples_affected, list):
        condition_samples = samples_affected
    elif isinstance(samples_affected, (int, float)):
        if isinstance(samples_affected, float) and samples_affected <= 1.0:
            samples_affected = int(len(condition_samples) * samples_affected)
        condition_samples = rng.choice(condition_samples, size=int(samples_affected))
    condition_samples = set(condition_samples)
    if isinstance(amount_affected, float):
        amount_affected = int(len(dataset.molecules[molecule]) * amount_affected)
    condition_affected_mols = rng.choice(
        dataset.molecules[molecule].index,
        size=amount_affected,
        replace=False,
    )
    factor = rng.normal(
            loc=log2_cond_factor_mean,
            scale=log2_cond_factor_std,
            size=len(condition_affected_mols),
        )
    factor = 2**factor
    for sample_name in dataset.sample_names:
        sample = dataset[sample_name]
        vals = sample.values[molecule][column].copy()
        if sample_name in condition_samples:
            vals.loc[condition_affected_mols] *= factor
        sample.values[molecule][result_column] = vals
    return dataset
