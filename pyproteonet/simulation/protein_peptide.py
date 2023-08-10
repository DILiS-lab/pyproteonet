from typing import List, Union, Optional, Iterable

from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset
from .sampling import draw_normal_log_space
from .modification import per_molecule_random_scaling, introduce_random_condition
from .random_error import multiply_exponential_gaussian, add_positive_gaussian, poisson_error
from ..processing.aggregation import neighbor_sum


def simulate_protein_peptide_dataset(
    molecule_set: MoleculeSet,
    samples: Union[int, List[str]] = 10,
    log_abundance_mu: float = 10,
    log_abundance_sigma: float = 2,
    log_protein_error_sigma: float = 0.3,
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
    mapping: str = "protein",
    random_seed: Optional[Union[int, float]] = None,
) -> Dataset:
    dataset = draw_normal_log_space(
        molecule_set=molecule_set,
        log_mu=log_abundance_mu,
        log_sigma=log_abundance_sigma,
        samples=samples,
        molecule=protein_molecule,
        column=protein_column,
        random_seed=random_seed,
    )
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
            random_seed=random_seed,
        )
    ground_truth_prot_vals = dataset.values[protein_molecule][protein_column]
    multiply_exponential_gaussian(
        dataset,
        molecule=protein_molecule,
        column=protein_column,
        sigma=log_protein_error_sigma,
        inplace=True,
        random_seed=random_seed,
    )
    neighbor_sum(
        dataset,
        input_molecule=protein_molecule,
        column=protein_column,
        mapping=mapping,
        result_molecule=peptide_molecule,
        result_column=peptide_column,
        only_unique=False,
        inplace=True,
    )
    if simulate_flyability:
        per_molecule_random_scaling(
            dataset=dataset,
            beta_distr_alpha=flyability_alpha,
            beta_distr_beta=flyability_beta,
            molecule=peptide_molecule,
            column=peptide_column,
            inplace=True,
            random_seed=random_seed,
        )
    add_positive_gaussian(
        dataset,
        molecule=peptide_molecule,
        column=peptide_column,
        mu=peptide_noise_mu,
        sigma=peptide_noise_sigma,
        inplace=True,
        random_seed=random_seed,
    )
    if peptide_poisson_error:
        poisson_error(
            dataset=dataset, molecule=peptide_molecule, column=peptide_column, random_seed=random_seed, inplace=True
        )
    dataset.values[protein_molecule][protein_column] = ground_truth_prot_vals
    return dataset
