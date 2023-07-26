import json
import os

import dgl
import torch
import numpy as np
import pandas as pd
import scipy
import networkx as nx
import psutil
import torch.nn.functional as F
from matplotlib import pyplot as plt
import seaborn as sns

from ..data.dataset import Dataset
from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet

PROTEIN_AGGREGATION_METHODS = {
        'sum': lambda x: x.groupby('protein').sum(),
        'mean': lambda x: x.groupby('protein').mean(),
        'max': lambda x: x.groupby('protein').max(),
        'top3' :lambda x: pd.DataFrame(x.groupby('protein')['abundance'].nlargest(3).groupby('protein').mean())
}

def simulate_peptide_based(aggregation_methods = ['sum','mean', 'max'], num_peptides = 1000, num_proteins = 100, num_samples_per_group = 10,
                           fraction_affected_by_condition = 0.2,
                           log_abundance_mean = 1.5, log_abundance_std = 0.5, condition_group_difference_means = [0, 1.5], 
                           condition_group_difference_stds = [0, 0.5], error_mean = 0, error_std = 0.5,
                           connection_probabilities = [1, 0.5, 0.25],
                           random_seed = None):
    rng = np.random.default_rng(random_seed)
    log_peptide_abundance = rng.normal(loc=log_abundance_mean, scale=log_abundance_std, size=num_peptides)
    peptide_abundance = np.exp(log_peptide_abundance)
    condition_affected_peptides = rng.choice(num_peptides, size=int(num_peptides * fraction_affected_by_condition), replace=False)
    #peptide_protein_index = rng.choice(num_proteins, size=num_peptides, replace=True)
    #Simulate peptide to protein correspondences
    protein_partners = rng.random((num_peptides, num_proteins)).argsort(axis=1)[:,:len(connection_probabilities)]
    a,b = [], []
    for i, cp in enumerate(connection_probabilities):
        choices = rng.choice(num_peptides, size=int(num_peptides * cp), replace=False)
        a.append(choices)
        b.append(protein_partners[choices,np.full(choices.shape[0], i)])
    a,b = np.concatenate(a), np.concatenate(b)
    peptide_gene_mapping = pd.DataFrame({'id':a, 'gene':b})
    protein_ids = np.arange(num_proteins)
    protein_gene_mapping = pd.DataFrame({'id':protein_ids, 'gene':protein_ids})
    peptide_ids = pd.DataFrame(index=np.arange(num_peptides))
    protein_ids = pd.DataFrame(index=protein_ids)
    molecule_set = MoleculeSet(molecules={'protein':protein_ids, 'peptide':peptide_ids},
                               mappings={'gene': {'protein':protein_gene_mapping, 'peptide':peptide_gene_mapping}})
    data_sets = dict()
    for method in aggregation_methods:
        method_data_sets = Dataset(molecule_set=molecule_set)
        for k in range(len(condition_group_difference_means)):
            condition_effect = np.zeros(num_peptides)
            effect = rng.normal(loc=condition_group_difference_means[k], scale=condition_group_difference_stds[k],
                                      size=condition_affected_peptides.shape[0])
            condition_effect[condition_affected_peptides] += effect
            for j in range(num_samples_per_group):
                values = {
                    'peptide': {},
                    'protein': {}
                }
                #Simulate peptide abbundances
                error = rng.normal(loc=error_mean, scale=error_std, size=num_peptides)
                peptides = peptide_abundance + condition_effect + error
                peptides = pd.DataFrame({'abundance':peptides})#, 'protein':peptide_protein_index})
                #Compute protein abbundances
                #protein aggregation methods expect "protein" column so we rename the gene column 
                #this (works because our data has 1:1 gene-protein correspondance)
                data = peptide_gene_mapping.rename(columns={'gene':'protein'})
                data['abundance'] = peptides.loc[peptide_gene_mapping.id, 'abundance'].to_numpy()
                proteins = PROTEIN_AGGREGATION_METHODS[method](data)
                sample_name = f'condition{k}_sample{j}'
                values['peptide'] = peptides
                values['protein'] = proteins
                method_data_sets.create_sample(name=sample_name, values=values)
                #method_data_sets[sample_name] = DatasetSample(molecule_set=molecule_set, values=values)
        data_sets[method] = method_data_sets
    return data_sets