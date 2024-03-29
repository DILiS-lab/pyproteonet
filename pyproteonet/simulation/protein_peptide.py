from typing import List, Union, Optional, Iterable

import numpy as np

from .condition import introduce_random_condition

from .utils import get_numpy_random_generator
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset
from .sampling import draw_normal_log_space
from .modification import per_molecule_random_scaling
from .random_error import multiply_exponential_gaussian, add_positive_gaussian, poisson_error
from pyproteonet.aggregation.partner_summarization import partner_aggregation


def simulate_protein_peptide_dataset(
    molecule_set: MoleculeSet,
    mapping: str,
    samples: Union[int, List[str]] = 10,
    log_abundance_mu: float = 10,
    log_abundance_sigma: float = 2,
    log_protein_error_sigma: float = 0.3,
    log_peptide_error_sigma: float = 0,
    simulate_flyability: bool = True,
    flyability_alpha: float = 5,
    flyability_beta: float = 2.5,
    peptide_noise_mu: float = 0,
    peptide_noise_sigma: float = 100,
    peptide_poisson_error: bool = True,
    condition_samples: List[Union[float, List[str]]] = [],
    condition_affected: List[Union[float, int, Iterable]] = [],
    log2_condition_means: List[float] = [],
    log2_condition_stds: List[float] = [],
    protein_column: str = "abundance_gt",
    peptide_column: str = "abundance",
    protein_molecule: str = "protein",
    peptide_molecule: str = "peptide",
    calculate_peptide_gt: bool = True,
    peptide_gt_column: Optional[str] = None,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
    print_parameters: bool = False,
) -> Dataset:
    """High-level wrapper for the simulation of a protein-pepide dataset wrapping multiple simulation steps.

    Details about single steps can be found in the corresponding simulation functions.

    Args:
        molecule_set (MoleculeSet): Underlying molecule set specifying the proteins and peptides as well as the mapping between them.
        mapping (str): MoleculeSet mapping that defines the protein to peptide relation.
        samples (Union[int, List[str]], optional): How many samples to generate. Defaults to 10.
        log_abundance_mu (float, optional): Mean of protein abundance in log space. Defaults to 10.
        log_abundance_sigma (float, optional): Standard deviation of protein abundance in log space. Defaults to 2.
        log_protein_error_sigma (float, optional): Standard deviation of normal distributed, zero centered protein error in log space. Defaults to 0.3.
         log_peptide_error_sigma (float, optional): Standard deviation of 0 centered, normal peptide error in log space. Defaults to 0.
        simulate_flyability (bool, optional): Whether every peptide should have a simulated flyability,
            defining which fraction of the peptide abundance is measured. Defaults to True.
        flyability_alpha (float, optional): Alpha parameter of beta distribution used to sample peptide flyability values. Defaults to 5.
        flyability_beta (float, optional): Beta prameter of beta distribution used to sample peptide flyability values. Defaults to 2.5.
        peptide_noise_mu (float, optional): Mean of normal distributed positive noise term to peptide abundances.
            To assure the noise is always positive the absolute value of the value sampled from the normal distribution is taken.
            Attention, NOT in log space. Defaults to 0.
        peptide_noise_sigma (float, optional): Standard deviation of positive peptide noise normal distribution (see above).
            Attention, NOT in log space. Defaults to 100.
        peptide_poisson_error (bool, optional): Whether to sample the final peptide abundance from a poisson distribution 
            centered at the computed abundance value to simulate counting effects when measuring peptides. Defaults to True.
        condition_samples (List[Union[float, List[str]]], optional): Condition groups. If given as list of values in the [0,1] interval
            every value describes the fraction of samples affected by the condition. If given as list of lists of strings, every list of strings
            represents a condition group and the strings are the names of the samples affected by this condition group. Defaults to [].
        condition_affected (List[Union[float, int, Iterable]], optional): List of condition affected proteins. If a list of floats/ints is given
            every value is interpreted as the fraction/absolute number of condition affected proteins and those are sampled randomly. If a list
            of iterables if given (e.g. a list of pandas Series) every iterable is interpreted as the protein indices of proteins affected in the
            corresponding condition group. Defaults to [].
        log2_condition_means (List[float], optional): List of mean values for the condition factor distributions for each condition group.
            Defaults to [].
        log2_condition_stds (List[float], optional): List of standard deviatoin values for the condition factor distributions for each condition group.
            Defaults to [].
        protein_column (str, optional): Column to write ground truth protein values to. Defaults to "abundance_gt".
        peptide_column (str, optional): Column to write peptide values to. Defaults to "abundance".
        protein_molecule (str, optional): Molecule name used for protein molecule type in the MoleculeSet. Defaults to "protein".
        peptide_molecule (str, optional): Molecule name of the peptide molecule type in the MoleculeSet. Defaults to "peptide".
        random_seed (Optional[Union[int, float]], optional): Random seed to use for random value generation. Defaults to None.

    Returns:
        Dataset: _description_
    """    
    #Use different seeds for different step because otherwise the exact same values are drawn in multiple steps which can lead to
    # strange effects (e.g. the product of two normal draws would always be positive because the same value is drawn twice). 
    if print_parameters:
        print(f"Protein distribution: mean(log_e)={log_abundance_mu}, std(log_e)={log_abundance_sigma}")
        print(f"Protein error std (log_e):{log_protein_error_sigma}")
        print(f"Flyability: {simulate_flyability}")
        if simulate_flyability:
            print(f"Flyability: alpha:{flyability_alpha}, beta:{flyability_beta}")
        print(f"Petide poisson errror: {peptide_poisson_error}")
        print(f"Peptide noise: mu(log_e):{peptide_noise_mu}, sigma(log_e):{peptide_noise_sigma}")
        print(f"Condition_samples:{condition_samples}, condition_affeced:{condition_affected},")
        print(f"condition_means(log2):{log2_condition_means}, condition_sigmas(log2){log2_condition_stds}")
    seed = get_numpy_random_generator(seed=random_seed)
    dataset = draw_normal_log_space(
        molecule_set=molecule_set,
        log_mu=log_abundance_mu,
        log_sigma=log_abundance_sigma,
        samples=samples,
        molecule=protein_molecule,
        column=protein_column,
        random_seed=seed,
    )
    if print_parameters:
        print(f"Number samples: {len(dataset.samples_dict)}")
    for samples, affected, mean, std in zip(condition_samples, condition_affected, log2_condition_means, log2_condition_stds):
        introduce_random_condition(
            dataset,
            molecule=protein_molecule,
            column=protein_column,
            inplace=True,
            affected=affected,
            log2_cond_factor_mean=mean,
            log2_cond_factor_std=std,
            samples=samples,
            random_seed=seed,
        )
    ground_truth_prot_vals = dataset.values[protein_molecule][protein_column]
    multiply_exponential_gaussian(
        dataset,
        molecule=protein_molecule,
        column=protein_column,
        sigma=log_protein_error_sigma,
        inplace=True,
        random_seed=seed,
    )
    partner_aggregation(
        dataset=dataset,
        molecule=peptide_molecule,
        partner_column=protein_column,
        mapping=mapping,
        method="sum",
        result_column=peptide_column,
        only_unique=False,
    )
    if log_peptide_error_sigma != 0:
         multiply_exponential_gaussian(
            dataset,
            molecule=peptide_molecule,
            column=peptide_column,
            sigma=log_peptide_error_sigma,
            inplace=True,
            random_seed=seed,
        )
    if simulate_flyability:
        per_molecule_random_scaling(
            dataset=dataset,
            beta_distr_alpha=flyability_alpha,
            beta_distr_beta=flyability_beta,
            molecule=peptide_molecule,
            column=peptide_column,
            inplace=True,
            random_seed=seed,
        )
    add_positive_gaussian(
        dataset,
        molecule=peptide_molecule,
        column=peptide_column,
        mu=peptide_noise_mu,
        sigma=peptide_noise_sigma,
        inplace=True,
        random_seed=seed,
    )
    if peptide_poisson_error:
        poisson_error(
            dataset=dataset, molecule=peptide_molecule, column=peptide_column, random_seed=seed, inplace=True
        )
    dataset.values[protein_molecule][protein_column] = ground_truth_prot_vals
    if calculate_peptide_gt:
        if peptide_gt_column is None:
            peptide_gt_column = peptide_column + '_gt'
        dataset.values[peptide_molecule][peptide_gt_column] = partner_aggregation(dataset=dataset, molecule=peptide_molecule, mapping=mapping,
                                                                                   partner_column=protein_column, method='sum', only_unique=False)
    return dataset
