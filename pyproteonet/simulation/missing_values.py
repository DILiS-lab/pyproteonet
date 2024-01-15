from typing import Optional, Union

import numpy as np
from numpy.random import Generator

from .utils import get_numpy_random_generator
from ..data.dataset import Dataset
from ..data.dataset_sample import DatasetSample


def simulate_mnars_thresholding_sample(
    sample: DatasetSample,
    thresh_mean: float,
    thresh_std: float,
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    in_log_space: bool = False,
    rng: Optional[Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()
    mask = sample.non_missing_mask(molecule=molecule, column=column)
    thresh = rng.normal(loc=thresh_mean, scale=thresh_std, size=mask.sum())
    vals = sample.values[molecule].loc[mask, column]
    if in_log_space:
        vals = np.log(vals)
    mask = mask & (vals < thresh)
    if result_column is None:
        result_column = column
    else:
        sample.values[molecule][result_column] = sample.values[molecule][column].copy()
    sample.values[molecule].loc[mask, result_column] = sample.missing_value
    if mask_column is not None:
        sample.values[molecule].loc[:, mask_column] = False
        sample.values[molecule].loc[mask, mask_column] = True
    return sample


def simulate_mnars_thresholding(
    dataset: Dataset,
    thresh_mu: float,
    thresh_sigma: float,
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    in_log_space: bool = False,
    rng: Optional[Generator] = None,
    inplace: bool = False,
)->Dataset:
    """Simulates (missing not at random) MNAR missing values.
      For each value a threshold is drawn from a normal distribution. Values below the threshold are masked.

    Args:
        dataset (Dataset): Dataset to simulate missing values for.
        thresh_mu (float): The mean of the normal distribution to draw the thresholds from.
        thresh_sigma (float): The standard deviation of the normal distribution to draw the thresholds from.
        molecule (str, optional): The molecule to apply missing values to. Defaults to "peptide".
        column (str, optional): The value column to apply missing values to. Defaults to "abundance".
        result_column (Optional[str], optional): Value column to write the results to, if not given same as column . Defaults to None.
        mask_column (Optional[str], optional): If given, the name of the value column to store in whether a value got masked or not. Defaults to None.
        in_log_space (bool, optional): Whether to loagrithmize abundance values before doing the thresholding. Defaults to False.
        rng (Optional[Generator], optional): The random generator to use to draw missing values. Defaults to None.
        inplace (bool, optional): If false the dataset will be copied before simulating missing values. Defaults to False.

    Returns:
        Dataset: The dataset with simulated missing values.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not inplace:
        dataset = dataset.copy()
    return dataset.sample_apply(
        fn=simulate_mnars_thresholding_sample,
        thresh_mean=thresh_mu,
        thresh_std=thresh_sigma,
        molecule=molecule,
        column=column,
        result_column=result_column,
        mask_column=mask_column,
        in_log_space=in_log_space,
        rng=rng,
    )


def simulate_mcars_sample(
    sample: DatasetSample,
    amount: Union[int, float],
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    mask_only_non_missing: bool = False, 
    rng: Optional[Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()
    num_vals = sample.values[molecule].shape[0]
    if isinstance(amount, float) and amount <= 1.0:
        amount = int(num_vals * amount)
    else:
        amount = int(amount)
    mask = sample.non_missing_mask(molecule=molecule, column=column)
    num_vals = mask.sum()
    if amount > num_vals:
        raise ValueError(f"Only {num_vals} non missing values are in the sample but {amount} MCARS were requests!")
    if mask_only_non_missing:
        mcar_mask = rng.choice(np.nonzero(mask.values)[0], size=amount, replace=False)
    else:
        mcar_mask = rng.choice(mask.shape[0], size=amount, replace=False)
    mask[:] = False
    mask.iloc[mcar_mask] = True
    if result_column is None:
        result_column = column
    else:
        sample.values[molecule][result_column] = sample.values[molecule][column].copy()
    sample.values[molecule].loc[mask, result_column] = sample.missing_value
    if mask_column is not None:
        sample.values[molecule].loc[:, mask_column] = False
        sample.values[molecule].loc[mask, mask_column] = True
    return sample


def simulate_mcars(
    dataset: Dataset,
    amount: Union[int, float],
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    mask_column: Optional[str] = None,
    rng: Optional[Generator] = None,
    mask_only_non_missing: bool = True, 
    inplace: bool = False,
):
    
    """Simulates (missing at random) MCAR missing values, by masking a given amount of values at random.
    Args:
        dataset (Dataset): Dataset to simulate missing values for.
        amount (Union[int, float]): The amount of values to mask. If float between 0 and 1, the amount is interpreted as a fraction of the total number of values.
        molecule (str, optional): The molecule to apply missing values to. Defaults to "peptide".
        column (str, optional): The value column to apply missing values to. Defaults to "abundance".
        result_column (Optional[str], optional): Value column to write the results to, if not given same as column . Defaults to None.
        mask_column (Optional[str], optional): If given, the name of the value column to store in whether a value got masked or not. Defaults to None.
        rng (Optional[Generator], optional): The random generator to use to draw missing values. Defaults to None.
        mask_only_non_missing (bool, optional): Whether to only mask non missing values. Defaults to True.
        inplace (bool, optional): If false the dataset will be copied before simulating missing values. Defaults to False.

    Returns:
        Dataset: The dataset with simulated missing values.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not inplace:
        dataset = dataset.copy()
    return dataset.sample_apply(
        simulate_mcars_sample,
        amount=amount,
        molecule=molecule,
        column=column,
        result_column=result_column,
        mask_column=mask_column,
        mask_only_non_missing=mask_only_non_missing,
        rng=rng,
    )


def simulate_mnars_mcars_lazar_sample(
    sample: DatasetSample,
    alpha=0.5,
    beta=0.8,
    mnar_thresh_std=0.01,
    missing_value=np.nan,
    rng=None,
    use_log_space: bool = False,
):
    if rng is None:
        rng = np.random.default_rng()
    sample = sample.copy()
    peptide_values = sample.values["peptide"]
    peptides = peptide_values.abundance.to_numpy()
    if use_log_space:
        # assert (peptides > 0).all()
        peptides = np.log(peptides)
    num_peptide_measures = peptides.shape[0]
    mnar_thresh_mean = np.quantile(peptides, alpha)
    mnar_thresh = rng.normal(loc=mnar_thresh_mean, scale=mnar_thresh_std, size=num_peptide_measures)
    mnar_mask = peptides < mnar_thresh
    mnar_bernoulli = rng.binomial(n=1, p=beta * alpha, size=peptides.shape[0]).astype(bool)
    mnar_mask = mnar_mask & mnar_bernoulli
    num_mcars = int((1 - beta) * alpha * num_peptide_measures)
    mcar_mask = np.zeros(peptides.shape[0], dtype=bool)
    mcar_mask[rng.choice(num_peptide_measures, size=num_mcars, replace=False)] = 1
    peptide_values["is_mcar"] = mcar_mask
    peptide_values["is_mnar"] = mnar_mask
    peptide_values.loc[mcar_mask, "abundance"] = missing_value
    peptide_values.loc[mnar_mask, "abundance"] = missing_value
    return sample


def simulate_mnars_mcars_lazar(
    dataset: Dataset,
    alpha=0.5,
    beta=0.8,
    mnar_thresh_std=0.01,
    missing_value=0,
    use_log_space: bool = False,
    random_seed=None,
    inplace: bool = False,
):
    rng = np.random.default_rng(random_seed)
    dataset.missing_value = missing_value
    if not inplace:
        dataset = dataset.copy()
    return dataset.sample_apply(
        fn=simulate_mnars_mcars_lazar_sample,
        alpha=alpha,
        beta=beta,
        mnar_thresh_std=mnar_thresh_std,
        missing_value=missing_value,
        use_log_space=use_log_space,
        rng=rng,
    )
