from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

from .utils import get_numpy_random_generator
from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


def simulate_protein_based_log_space(
    molecule_set: MoleculeSet,
    num_samples_per_group=10,
    fraction_affected_by_condition=0.2,
    log_abundance_mean=10,
    log_abundance_std=2,
    log_condition_group_difference_means=[0, 1.5],
    log_condition_group_difference_stds=[0, 0.5],
    protein_error_std_multiplier=0.5,
    peptide_error_std=0.5,
    error_in_log_space: bool = False,
    simulate_flyability: bool = True,
    flyability_alpha: float = 5,
    flyability_beta: float = 2.5,
    peptide_abundance_factor: float = 1,
    random_seed=None,
    protein_result_column: str = "abundance",
    peptide_result_column: str = "abundance",
    mapping="protein",
    *args,
    **kwargs,
):
    rng = get_numpy_random_generator(seed=random_seed)
    if "protein" not in molecule_set.molecules:
        raise ValueError('Molecule set must contain key "protein".')
    if "peptide" not in molecule_set.molecules:
        raise ValueError('Molecule set must contain key "peptide".')
    if mapping not in molecule_set.mappings:
        raise ValueError(
            f"Molecule set must contain a {mapping} mapping to relate peptides with proteins."
        )

    num_proteins = len(molecule_set.molecules["protein"])
    num_peptides = len(molecule_set.molecules["peptide"])
    log_protein_abundance = rng.normal(
        loc=log_abundance_mean, scale=log_abundance_std, size=num_proteins
    )
    if simulate_flyability:
        peptide_flyabilities = scipy.stats.beta.rvs(
            a=flyability_alpha, b=flyability_beta, size=num_peptides, random_state=rng
        )
    else:
        peptide_flyabilities = np.ones(num_peptides)
    peptide_flyabilities = pd.Series(
        peptide_flyabilities, index=molecule_set.molecules["peptide"].index
    )
    condition_affected_protein_numbers = rng.choice(
        molecule_set.molecules["protein"].shape[0],
        size=int(num_proteins * fraction_affected_by_condition),
        replace=False,
    )
    condition_affected_proteins = (
        molecule_set.molecules["protein"].iloc[condition_affected_protein_numbers].index
    )
    molecule_set.molecules["protein"]["condition_affected"] = False
    molecule_set.molecules["protein"].loc[
        condition_affected_proteins, "condition_affected"
    ] = True

    peptide_protein_mapping = molecule_set.get_mapped_pairs(
        molecule_a="protein", molecule_b="peptide", mapping=mapping
    )

    dataset = Dataset(molecule_set=molecule_set)
    for k in range(len(log_condition_group_difference_means)):
        condition_effect = np.zeros(num_proteins)
        effect = rng.normal(
            loc=log_condition_group_difference_means[k],
            scale=log_condition_group_difference_stds[k],
            size=condition_affected_protein_numbers.shape[0],
        )
        condition_effect[condition_affected_protein_numbers] += effect
        for j in range(num_samples_per_group):
            # Simulate peptide abbundances
            protein_abundance = log_protein_abundance + condition_effect
            protein_abundance = np.exp(protein_abundance)
            protein_error = rng.normal(
                loc=0, scale=protein_error_std_multiplier * (np.log(protein_abundance) if error_in_log_space else protein_abundance)
            )
            protein_abundance += protein_error
            protein_abundance = np.clip(a=protein_abundance, a_min=0, a_max=None)
            proteins = pd.DataFrame(
                {protein_result_column: protein_abundance},
                index=molecule_set.molecules["protein"].index,
            )
            # Compute protein abbundances
            # protein aggregation methods expect "protein" column so we rename the gene column
            # (this works because our data has 1:1 gene-protein correspondance)
            peptide_abundances = peptide_protein_mapping.copy()
            peptide_abundances[peptide_result_column] = proteins.loc[
                peptide_abundances["protein"], protein_result_column
            ].to_numpy()
            peptide_abundances = peptide_abundances.groupby("peptide")[peptide_result_column].sum()

            peptide_abundances = peptide_abundances * peptide_flyabilities.loc[peptide_abundances.index]
            peptide_abundances *= peptide_abundance_factor
            # peptide_error = rng.normal(
            #     loc=0, scale= peptide_error_std_multiplier * (np.log(peptide_abundances) if error_in_log_space else peptide_abundances)
            # )
            peptide_error = rng.normal(
                loc=0, scale= peptide_error_std
            )
            peptide_abundances += np.abs(peptide_error)
            peptide_abundances = peptide_abundances.clip(lower=0, upper=None)
            peptide_abundances.loc[:] = rng.poisson(lam=peptide_abundances.to_numpy())
            peptide_abundances = pd.DataFrame(
                {peptide_result_column:peptide_abundances},
                index=molecule_set.molecules["peptide"].index,
            )
            sample_name = f"condition{k}_sample{j}"
            values = {"peptide": peptide_abundances, "protein": proteins}
            dataset.create_sample(name=sample_name, values=values)
    return dataset
