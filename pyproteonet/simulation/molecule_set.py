from typing import List, Optional
import math

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import scipy

from .utils import get_numpy_random_generator
from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet
from ..data.dataset import Dataset

def molecule_set_from_degree_distribution(molecule1_degree_distribution: List[int] = [2, 5, 7, 7], molecule2_degree_distribution: List[int] = [1, 20, 10],
                                          molecule1_name: str = 'protein', molecule2_name: str = 'peptide',
                                          mapping_name: str = 'peptide-protein', random_seed: Optional[int] = None)->MoleculeSet:
    rng = get_numpy_random_generator(seed=random_seed)
    molecule1_degree_distribution = np.array(molecule1_degree_distribution).astype(int)
    molecule2_degree_distribution = np.array(molecule2_degree_distribution).astype(int)
    num_m1_edges = (np.arange(len(molecule1_degree_distribution)) * molecule1_degree_distribution).sum()
    num_m2_edges = (np.arange(len(molecule2_degree_distribution)) * molecule2_degree_distribution).sum()
    if num_m1_edges != num_m2_edges:
        raise ValueError("Sum of molecule1 degrees must match sum of molecule2 degrees")
    m1_degs = np.repeat(np.arange(len(molecule1_degree_distribution)), molecule1_degree_distribution)
    m2_degs = np.repeat(np.arange(len(molecule2_degree_distribution)), molecule2_degree_distribution)
    m1_ids = np.repeat(np.arange(len(m1_degs)), m1_degs)
    m2_ids = np.repeat(np.arange(len(m2_degs)), m2_degs)
    edges = np.stack((m1_ids, m2_ids), axis=1)
    while True:
        unique, ids, counts = np.unique(edges, axis=0, return_counts=True, return_inverse=True)
        mask = counts > 1
        if not mask.any():
            break
        duplicates = np.repeat(unique[mask], counts[mask] - 1, axis=0)
        rewired = []
        for dup in duplicates:
            id1 = rng.integers(unique.shape[0])
            id2 = rng.integers(2)
            dup[id2], unique[id1, id2] = unique[id1, id2], dup[id2]
            rewired.append(dup)
        edges = np.concatenate((unique, rewired), axis=0)
    m1s = pd.DataFrame(index=np.arange(m1_ids.max()+1))
    m2s = pd.DataFrame(index=np.arange(m2_ids.max()+1))
    mapping = pd.DataFrame({molecule2_name:edges[:,1], molecule1_name:edges[:,0]})
    mapping.set_index([molecule2_name, molecule1_name], inplace=True)
    ms = MoleculeSet(molecules = {molecule1_name:m1s, molecule2_name:m2s},
                     mappings = {mapping_name: mapping}
                    )
    return ms

def _relative_to_absolute_node_degrees(relative_node_degrees: List[float], num_nodes: int)->List[int]:
    assert abs(sum(relative_node_degrees) * num_nodes - num_nodes) < 1.0 #relative_node_degrees sum up to one (neglecting floating point inaccuracies)
    nodes_with_degrees = []
    slack = 0
    for rnd in relative_node_degrees[::-1]:
        nwd = rnd * num_nodes + slack
        slack = nwd - round(nwd)
        nodes_with_degrees.append(int(round(nwd)))
    return nodes_with_degrees[::-1]

def _degree_distribution_to_integers(node_degrees: List[float])->List[int]:
    result = []
    slack = 0
    for deg, num_nodes in enumerate(node_degrees):
        upper = math.ceil(num_nodes)
        diff = (upper - num_nodes)*deg
        if diff <= slack:
            result.append(upper)
            slack -= diff
        else:
            lower = math.floor(num_nodes)
            result.append(lower)
            slack += (num_nodes - lower)*deg
    return result

def simulate_molecule_set_protein_peptide(num_peptides = 1000, num_proteins = 100, 
                                          relative_peptide_node_degrees = [0.25, 0.5, 0.25],
                                          relative_protein_node_degrees = None,
                                          random_seed = None, *args, **kwargs):
    rng = get_numpy_random_generator(seed=random_seed)
    #Simulate peptide to protein correspondences
    peptide_node_degrees = _relative_to_absolute_node_degrees(relative_peptide_node_degrees, num_peptides)
    protein_deg_distribution = None
    b_nodes_deg0 = 0
    if relative_protein_node_degrees is not None:
        num_edges = ((np.arange(len(peptide_node_degrees))) * peptide_node_degrees).sum()
        #protein_deg_distribution = np.array(_relative_to_absolute_node_degrees(relative_protein_node_degrees, num_proteins))
        protein_deg_distribution = np.array(relative_protein_node_degrees)
        protein_deg_distribution /= protein_deg_distribution.sum()
        protein_deg_distribution *= num_edges / (protein_deg_distribution * np.arange(len(protein_deg_distribution))).sum()
        b_nodes_deg0 = round(protein_deg_distribution[0])
        protein_deg_distribution = np.array(_degree_distribution_to_integers(protein_deg_distribution), dtype=int)
        protein_deg_distribution = np.repeat(np.arange(len(protein_deg_distribution)), protein_deg_distribution).astype(float)
        #protein_deg_distribution *= num_edges / protein_deg_distribution.sum()
        rng.shuffle(protein_deg_distribution)
    a_b = []
    start = 0
    a_nodes = []
    a_nodes_deg0 = []
    for deg, nwd in enumerate(peptide_node_degrees):
        nodes = np.arange(start, start + nwd)
        if deg==0:
            a_nodes_deg0.append(nodes)
        else:
            a_nodes.append(np.stack([nodes, np.full(nwd, deg)], axis=1))
        start += nwd
    a_nodes = np.concatenate(a_nodes, axis=0)
    rng.shuffle(a_nodes, axis=0)
    epsilon = 1 / protein_deg_distribution.sum() * 0.001
    for a, deg in tqdm(a_nodes):
        a = np.full(deg, a)
        if protein_deg_distribution is None:
            b = rng.choice(num_proteins, size=deg, replace=False)
        else:
            #b = np.argpartition(protein_deg_distribution, -deg)[-deg:]
            probs = protein_deg_distribution + epsilon
            probs[probs<0] = 0
            probs /= probs.sum()
            b = rng.choice(protein_deg_distribution.shape[0], p=probs, size=deg, replace=False)
            protein_deg_distribution[b]-=1
        if len(a)!=len(b):
            raise ValueError()
        a_b.append(np.stack([a, b], axis=1))
    a_b = np.concatenate(a_b, axis=0)
    #correspondences = pd.DataFrame({'peptide':a, 'protein':b})
    protein_ids = np.squeeze(np.unique(a_b[:,1]))
    start_deg_0 = protein_ids.max()+1
    protein_ids = np.concatenate([protein_ids, np.arange(start_deg_0, start_deg_0 + b_nodes_deg0)])
    protein_protein_mapping = pd.DataFrame({'id':protein_ids, 'map_id':protein_ids})
    protein_ids = pd.DataFrame(index=protein_ids)
    a_nodes_deg0 = np.concatenate(a_nodes_deg0, axis=0)
    peptide_ids = pd.DataFrame(index=np.unique(np.concatenate([np.squeeze(a_b[:,0]), a_nodes_deg0], axis=0)))
    peptide_protein_mapping = pd.DataFrame({'id':a_b[:,0], 'map_id':a_b[:,1]})
    protein_mapping = {'protein':protein_protein_mapping, 'peptide':peptide_protein_mapping}
    #we assume exactly one protein per gene
    molecule_set = MoleculeSet(molecules={'peptide':peptide_ids, 'protein':protein_ids},
                               mappings={'gene':protein_mapping, 'protein':protein_mapping})
    return molecule_set