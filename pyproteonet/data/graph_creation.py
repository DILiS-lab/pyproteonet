import itertools
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Iterable, Union
import logging

import pandas as pd
import numpy as np
import dgl
import torch

if TYPE_CHECKING:
    from .molecule_set import MoleculeSet
    from .dataset_sample import DatasetSample
    from .molecule_graph import MoleculeGraph

# NODE_TYPE_MAPPING = {
#     "peptide": 0,
#     "mRNA": 1,
#     "protein": 2,
# }
# NODE_TYPE_INVERSE_MAPPING = {v: k for k, v in NODE_TYPE_MAPPING.items()}

logger = logging.Logger("graph_creation")


def create_graph_nodes_edges(
    molecule_set: "MoleculeSet",
    mappings: List[str],
    make_bidirectional: bool = True,
    add_self_edges: bool = True,
    node_type_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, int]]:
    node_mapping = dict()
    offset = 0
    if node_type_mapping is None:
        node_type_mapping = {molecule: i for i,molecule in enumerate(molecule_set.molecules.keys())}
        logger.info(f"Created node type mapping: {node_type_mapping}")
    for key in molecule_set.molecules.keys():
        index = molecule_set.molecules[key].index
        size = index.shape[0]
        node_ids = np.arange(offset, offset + size)
        offset += size
        node_mapping[key] = pd.DataFrame({"node_id": node_ids}, index=index)
    edges = []
    for mapping in mappings:
        mapping = molecule_set.get_mapping(mapping_name=mapping)
        source_mol, destination_mol = mapping.mapping_molecules
        source_mol = node_mapping[source_mol].loc[mapping.df.index.get_level_values(0), "node_id"].to_numpy()
        destination_mol = node_mapping[destination_mol].loc[mapping.df.index.get_level_values(1), "node_id"].to_numpy()
        edges.append(pd.DataFrame({"source_node":source_mol, "destination_node":destination_mol}))
    edges = pd.concat(edges, ignore_index=True)
    if make_bidirectional:
        edges2 = pd.DataFrame({"source_node": edges.destination_node, "destination_node": edges.source_node})
        edges = pd.concat([edges, edges2], ignore_index=True)
    nodes = []
    for key in node_mapping.keys():
        nm = node_mapping[key].copy()
        nm["molecule_id"] = nm.index
        nm.set_index("node_id", inplace=True, verify_integrity=True)
        nm["type"] = node_type_mapping[key]
        nm["type"] = nm["type"].astype(int)
        nodes.append(nm)
        # assert ('node_id' not in a.columns and 'type' not in a.columns
        # a['node_id'] = node_mapping[key].loc[a.index, 'node_id']
        # a['type'] =
    nodes = pd.concat(nodes, verify_integrity=True)
    if add_self_edges:
        self_edges = pd.DataFrame({"source_node": np.arange(len(nodes)), "destination_node": np.arange(len(nodes))})
        edges = pd.concat([edges, self_edges], ignore_index=True)
    return nodes, edges, node_mapping, node_type_mapping
