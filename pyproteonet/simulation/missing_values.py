from typing import Optional, Union

import numpy as np
from numpy.random import Generator
from ..data.dataset import Dataset
from ..data.dataset_sample import DatasetSample


def simulate_mnars_thresholding_sample(
    sample: DatasetSample,
    thresh_mean: float,
    thresh_std: float,
    molecule: str = "peptide",
    column: str = "abundance",
    write_mask_column: Optional[str] = "mnar_mask",
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
    sample.values[molecule].loc[mask, column] = sample.missing_abundance_value
    if write_mask_column is not None:
        sample.values[molecule].loc[:, write_mask_column] = False
        sample.values[molecule].loc[mask, write_mask_column] = True
    return sample


def simulate_mnars_thresholding(
    dataset: Dataset,
    thresh_mean: float,
    thresh_std: float,
    molecule: str = "peptide",
    column: str = "abundance",
    write_mask_column: Optional[str] = "mnar_mask",
    in_log_space: bool = False,
    rng: Optional[Generator] = None,
    inplace: bool = False,
):
    if rng is None:
        rng = np.random.default_rng()
    if not inplace:
        dataset = dataset.copy()
    return dataset.apply(
        fn=simulate_mnars_thresholding_sample,
        thresh_mean=thresh_mean,
        thresh_std=thresh_std,
        molecule=molecule,
        column=column,
        in_log_space=in_log_space,
        rng=rng,
    )


def simulate_mcars_sample(
    sample: DatasetSample,
    amount_mcars: Union[int, float],
    molecule: str = "peptide",
    column: str = "abundance",
    write_mask_column: Optional[str] = "mcar_mask",
    rng: Optional[Generator] = None,
):
    if rng is None:
        rng = np.random.default_rng()
    num_vals = sample.values[molecule].shape[0]
    if isinstance(amount_mcars, float) and amount_mcars <= 1.0:
        amount_mcars = int(num_vals * amount_mcars)
    else:
        amount_mcars = int(amount_mcars)
    mask = sample.non_missing_mask(molecule=molecule, column=column)
    num_vals = mask.sum()
    if amount_mcars > num_vals:
        raise ValueError(f"Only {num_vals} non missing values are in the sample but {amount_mcars} MCARS were requests!")
    mcar_mask = rng.choice(np.nonzero(mask)[0], size=amount_mcars, replace=False)
    mask[:] = False
    mask[mcar_mask] = True
    #import pdb; pdb.set_trace()
    sample.values[molecule].loc[mask, column] = sample.missing_abundance_value
    if write_mask_column is not None:
        sample.values[molecule].loc[:, write_mask_column] = False
        sample.values[molecule].loc[mask, write_mask_column] = True
    return sample


def simulate_mcars(
    dataset: Dataset,
    amount_mcars: Union[int, float],
    molecule: str = "peptide",
    column: str = "abundance",
    write_mask_column: Optional[str] = "mcar_mask",
    rng: Optional[Generator] = None,
    inplace: bool = False,
):
    if rng is None:
        rng = np.random.default_rng()
    if not inplace:
        dataset = dataset.copy()
    return dataset.apply(
        simulate_mcars_sample,
        amount_mcars=amount_mcars,
        molecule=molecule,
        column=column,
        write_mask_column=write_mask_column,
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
    return dataset.apply(
        fn=simulate_mnars_mcars_lazar_sample,
        alpha=alpha,
        beta=beta,
        mnar_thresh_std=mnar_thresh_std,
        missing_value=missing_value,
        use_log_space=use_log_space,
        rng=rng,
    )
