from typing import List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset


def _relative_to_absolute_node_degrees(
    relative_node_degrees: List[float], num_nodes: int
) -> List[int]:
    assert (
        abs(sum(relative_node_degrees) * num_nodes - num_nodes) < 1.0
    )  # relative_node_degrees sum up to one (up to float inaccuracies)
    nodes_with_degrees = []
    slack = 0
    for rnd in relative_node_degrees[::-1]:
        nwd = rnd * num_nodes + slack
        slack = nwd - round(nwd)
        nodes_with_degrees.append(int(round(nwd)))
    return nodes_with_degrees[::-1]


def simulate_protein_based_log_space2(
    num_peptides=1000,
    num_proteins=100,
    num_samples_per_group=10,
    fraction_affected_by_condition=0.2,
    log_abundance_mean=10,
    log_abundance_std=2,
    log_condition_group_difference_means=[0, 1.5],
    log_condition_group_difference_stds=[0, 0.5],
    protein_error_std_multiplier=0.5,
    peptide_error_std_multiplier=0.5,
    relative_peptide_node_degrees=[0.25, 0.5, 0.25],
    peptide_abundance_factor: float = 1,
    flyability_alpha: float = 5,
    flyability_beta: float = 2.5,
    relative_protein_node_degrees=None,
    random_seed=None,
    *args,
    **kwargs,
):
    rng = np.random.default_rng(seed=random_seed)
    log_protein_abundance = rng.normal(
        loc=log_abundance_mean, scale=log_abundance_std, size=num_proteins
    )
    peptide_flyabilities = scipy.stats.beta.rvs(
        a=flyability_alpha, b=flyability_beta, size=num_peptides, random_state=rng
    )
    condition_affected_proteins = rng.choice(
        num_proteins,
        size=int(num_proteins * fraction_affected_by_condition),
        replace=False,
    )
    # peptide_protein_index = rng.choice(num_proteins, size=num_peptides, replace=True)
    # Simulate peptide to protein correspondences
    peptide_node_degrees = _relative_to_absolute_node_degrees(
        relative_peptide_node_degrees, num_peptides
    )
    # protein_partners = []
    # len_con = len(peptide_node_degrees)
    # for deg in range(num_peptides):
    #     protein_partners.append(rng.choice(num_proteins, size=len_con))
    # protein_partners = np.stack(protein_partners, axis=0)
    # protein_partners = rng.random((num_peptides, num_proteins)).argsort(axis=1)[:,:len(connection_probabilities)]
    protein_deg_distribution = None
    if relative_protein_node_degrees is not None:
        num_edges = (
            (np.arange(len(peptide_node_degrees))) * peptide_node_degrees
        ).sum()
        protein_deg_distribution = np.array(
            _relative_to_absolute_node_degrees(
                relative_protein_node_degrees, num_proteins
            )
        )
        protein_deg_distribution = np.repeat(
            np.arange(len(protein_deg_distribution)), protein_deg_distribution
        ).astype(float)
        protein_deg_distribution *= num_edges / protein_deg_distribution.sum()
        rng.shuffle(protein_deg_distribution)
    a_b = []
    start = 0
    for deg, nwd in enumerate(peptide_node_degrees):
        if deg == 0:
            continue
        # deg += 1 #no need any more, degrees start counting at 0
        a_nodes = np.arange(start, start + nwd)
        start += nwd
        for a in a_nodes:
            a = np.full(deg, a)
            if protein_deg_distribution is None:
                b = rng.choice(num_proteins, size=deg, replace=False)
            else:
                b = np.argpartition(protein_deg_distribution, -deg)[-deg:]
                protein_deg_distribution[b] -= 1
            if len(a) != len(b):
                import pdb

                pdb.set_trace()
            a_b.append(np.stack([a, b], axis=1))
    a_b = np.concatenate(a_b, axis=0)
    # correspondences = pd.DataFrame({'peptide':a, 'protein':b})
    peptide_gene_mapping = pd.DataFrame({"id": a_b[:, 0], "map_id": a_b[:, 1]})
    protein_ids = np.arange(num_proteins)
    protein_gene_mapping = pd.DataFrame({"id": protein_ids, "map_id": protein_ids})
    peptide_ids = pd.DataFrame(index=np.arange(num_peptides))
    protein_ids = pd.DataFrame(index=protein_ids)
    protein_mapping = {"protein": protein_gene_mapping, "peptide": peptide_gene_mapping}
    # we assume exactly one protein per gene
    molecule_set = MoleculeSet(
        molecules={"peptide": peptide_ids, "protein": protein_ids},
        mappings={"map_id": protein_mapping, "protein": protein_mapping},
    )
    molecule_set.molecules["protein"]["condition_affected"] = False
    molecule_set.molecules["protein"].loc[
        condition_affected_proteins, "condition_affected"
    ] = True

    dataset = Dataset(molecule_set=molecule_set)
    for k in range(len(log_condition_group_difference_means)):
        condition_effect = np.zeros(num_proteins)
        effect = rng.normal(
            loc=log_condition_group_difference_means[k],
            scale=log_condition_group_difference_stds[k],
            size=condition_affected_proteins.shape[0],
        )
        condition_effect[condition_affected_proteins] += effect
        for j in range(num_samples_per_group):
            # Simulate peptide abbundances
            protein_abundance = log_protein_abundance + condition_effect
            protein_error = rng.normal(
                loc=0, scale=protein_error_std_multiplier * 1 / protein_abundance
            )
            protein_abundance = np.exp(protein_abundance) + protein_error
            proteins = pd.DataFrame({"abundance": protein_abundance})
            # Compute protein abbundances
            # protein aggregation methods expect "protein" column so we rename the gene column
            # (this works because our data has 1:1 gene-protein correspondance)
            peptide_abundances = peptide_gene_mapping.copy()
            peptide_abundances["abundance"] = proteins.loc[
                peptide_abundances["map_id"], "abundance"
            ].to_numpy()
            peptide_abundances = pd.DataFrame(
                peptide_abundances.groupby("id")["abundance"].sum()
            )
            peptide_abundances["abundance"] = (
                peptide_abundances.abundance
                * peptide_flyabilities[peptide_abundances.index]
            )
            peptide_abundances *= peptide_abundance_factor
            peptide_error = rng.normal(
                loc=0, scale= peptide_error_std_multiplier / np.log(peptide_abundances)
            )
            peptide_abundances += peptide_error
            sample_name = f"condition{k}_sample{j}"
            values = {
                "peptide": peptide_abundances,
                "protein": proteins.reset_index(drop=True),
            }
            dataset.create_sample(name=sample_name, values=values)
    return dataset


def simulate_protein_based_log_space(
    num_peptides=1000,
    num_proteins=100,
    num_samples_per_group=10,
    fraction_affected_by_condition=0.2,
    log_abundance_mean=10,
    log_abundance_std=2,
    log_condition_group_difference_means=[0, 1.5],
    log_condition_group_difference_stds=[0, 0.5],
    log_error_mean=0,
    log_error_std=0.5,
    relative_peptide_node_degrees=[0.25, 0.5, 0.25],
    peptide_abundance_factor: float = 1,
    flyability_alpha: float = 5,
    flyability_beta: float = 2.5,
    relative_protein_node_degrees=None,
    random_seed=None,
    *args,
    **kwargs,
):
    rng = np.random.default_rng(seed=random_seed)
    log_protein_abundance = rng.normal(
        loc=log_abundance_mean, scale=log_abundance_std, size=num_proteins
    )
    peptide_flyabilities = scipy.stats.beta.rvs(
        a=flyability_alpha, b=flyability_beta, size=num_peptides, random_state=rng
    )
    condition_affected_proteins = rng.choice(
        num_proteins,
        size=int(num_proteins * fraction_affected_by_condition),
        replace=False,
    )
    # peptide_protein_index = rng.choice(num_proteins, size=num_peptides, replace=True)
    # Simulate peptide to protein correspondences
    peptide_node_degrees = _relative_to_absolute_node_degrees(
        relative_peptide_node_degrees, num_peptides
    )
    # protein_partners = []
    # len_con = len(peptide_node_degrees)
    # for deg in range(num_peptides):
    #     protein_partners.append(rng.choice(num_proteins, size=len_con))
    # protein_partners = np.stack(protein_partners, axis=0)
    # protein_partners = rng.random((num_peptides, num_proteins)).argsort(axis=1)[:,:len(connection_probabilities)]
    protein_deg_distribution = None
    if relative_protein_node_degrees is not None:
        num_edges = (
            (np.arange(len(peptide_node_degrees))) * peptide_node_degrees
        ).sum()
        protein_deg_distribution = np.array(
            _relative_to_absolute_node_degrees(
                relative_protein_node_degrees, num_proteins
            )
        )
        protein_deg_distribution = np.repeat(
            np.arange(len(protein_deg_distribution)), protein_deg_distribution
        ).astype(float)
        protein_deg_distribution *= num_edges / protein_deg_distribution.sum()
        rng.shuffle(protein_deg_distribution)
    a_b = []
    start = 0
    for deg, nwd in enumerate(peptide_node_degrees):
        if deg == 0:
            continue
        # deg += 1 #no need any more, degrees start counting at 0
        a_nodes = np.arange(start, start + nwd)
        start += nwd
        for a in a_nodes:
            a = np.full(deg, a)
            if protein_deg_distribution is None:
                b = rng.choice(num_proteins, size=deg, replace=False)
            else:
                b = np.argpartition(protein_deg_distribution, -deg)[-deg:]
                protein_deg_distribution[b] -= 1
            if len(a) != len(b):
                import pdb

                pdb.set_trace()
            a_b.append(np.stack([a, b], axis=1))
    a_b = np.concatenate(a_b, axis=0)
    # correspondences = pd.DataFrame({'peptide':a, 'protein':b})
    peptide_gene_mapping = pd.DataFrame({"id": a_b[:, 0], "map_id": a_b[:, 1]})
    protein_ids = np.arange(num_proteins)
    protein_gene_mapping = pd.DataFrame({"id": protein_ids, "map_id": protein_ids})
    peptide_ids = pd.DataFrame(index=np.arange(num_peptides))
    protein_ids = pd.DataFrame(index=protein_ids)
    protein_mapping = {"protein": protein_gene_mapping, "peptide": peptide_gene_mapping}
    # we assume exactly one protein per gene
    molecule_set = MoleculeSet(
        molecules={"peptide": peptide_ids, "protein": protein_ids},
        mappings={"map_id": protein_mapping, "protein": protein_mapping},
    )
    molecule_set.molecules["protein"]["condition_affected"] = False
    molecule_set.molecules["protein"].loc[
        condition_affected_proteins, "condition_affected"
    ] = True

    dataset = Dataset(molecule_set=molecule_set)
    for k in range(len(log_condition_group_difference_means)):
        condition_effect = np.zeros(num_proteins)
        effect = rng.normal(
            loc=log_condition_group_difference_means[k],
            scale=log_condition_group_difference_stds[k],
            size=condition_affected_proteins.shape[0],
        )
        condition_effect[condition_affected_proteins] += effect
        for j in range(num_samples_per_group):
            # Simulate peptide abbundances
            error = rng.normal(
                loc=log_error_mean, scale=log_error_std, size=num_proteins
            )
            protein_abundance = log_protein_abundance + condition_effect + error
            protein_abundance = np.exp(protein_abundance)
            proteins = pd.DataFrame({"abundance": protein_abundance})
            # Compute protein abbundances
            # protein aggregation methods expect "protein" column so we rename the gene column
            # (this works because our data has 1:1 gene-protein correspondance)
            peptide_abundances = peptide_gene_mapping.copy()
            peptide_abundances["abundance"] = proteins.loc[
                peptide_abundances["map_id"], "abundance"
            ].to_numpy()
            peptide_abundances = pd.DataFrame(
                peptide_abundances.groupby("id")["abundance"].sum()
            )
            peptide_abundances["abundance"] = (
                peptide_abundances.abundance
                * peptide_flyabilities[peptide_abundances.index]
            )
            peptide_abundances *= peptide_abundance_factor
            sample_name = f"condition{k}_sample{j}"
            values = {
                "peptide": peptide_abundances,
                "protein": proteins.reset_index(drop=True),
            }
            dataset.create_sample(name=sample_name, values=values)
    return dataset
