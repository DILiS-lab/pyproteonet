from typing import List, Union, Optional

from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset
from .sampling import draw_normal_log_space
from .modification import per_molecule_random_scaling, introduce_random_condition
from .random_error import multiply_exponential_gaussian, add_positive_gaussian, poisson_error
from ..processing.aggregation import neighbor_sum


def simulate_protein_peptide_dataset(
    molecule_set: MoleculeSet,
    num_samples: int = 10,
    log_abundance_mu: float = 10,
    log_abundance_sigma: float = 2,
    log_protein_error_std: float = 0.3,
    simulate_flyability: bool = True,
    flyability_alpha: float = 5,
    flyability_beta: float = 2.5,
    peptide_noise_sigma: float = 100,
    peptide_poisson_error: bool = True,
    condition_affected_samples: List[float]=0.0,
    fraction_affected_by_condition: List[float]=[],
    log2_condition_means: List[float]=[],
    log2_condition_stds: List[float]=[],
    protein_column: str = "abundance",
    peptide_column: str = "abundance",
    mapping: str = "protein",
    random_seed: Optional[Union[int, float]]=None,
) -> Dataset:
    dataset = draw_normal_log_space(
        molecule_set=molecule_set,
        log_mu=log_abundance_mu,
        log_sigma=log_abundance_sigma,
        num_samples=num_samples,
        molecule="protein",
        column=protein_column,
        random_seed=random_seed
    )
    condition_samples = list(dataset.sample_names)[-int(condition_affected_samples * len(dataset)) :]
    for affected, mean, std in zip(fraction_affected_by_condition, log2_condition_means, log2_condition_stds):
        introduce_random_condition(
            dataset,
            molecule="protein",
            column=protein_column,
            inplace=True,
            amount_affected=affected,
            log2_cond_factor_mean=mean,
            log2_cond_factor_std=std,
            samples_affected=condition_samples,
            random_seed=random_seed
        )
    multiply_exponential_gaussian(
        dataset,
        molecule="protein",
        column=protein_column,
        sigma=log_protein_error_std,
        inplace=True,
        random_seed=random_seed,
    )
    neighbor_sum(
        dataset,
        input_molecule="protein",
        input_column=protein_column,
        mapping=mapping,
        result_molecule="peptide",
        result_column=peptide_column,
        only_unique=False,
        inplace=True,
    )
    if simulate_flyability:
        per_molecule_random_scaling(
            dataset=dataset,
            beta_distr_alpha=flyability_alpha,
            beta_distr_beta=flyability_beta,
            molecule="peptide",
            column=peptide_column,
            inplace=True,
            random_seed=random_seed,
        )
    add_positive_gaussian(
        dataset,
        molecule="peptide",
        column=peptide_column,
        mu=0,
        sigma=peptide_noise_sigma,
        inplace=True,
        random_seed=random_seed,
    )
    if peptide_poisson_error:
        poisson_error(dataset=dataset, molecule='peptide', column=peptide_column, random_seed=random_seed, inplace=True)
    return dataset
