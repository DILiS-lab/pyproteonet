from dataclasses import dataclass
from typing import Optional, Dict

import pandas as pd
import numpy as np

from .molecule_graph import MoleculeGraph
from .dataset_sample import DatasetSample

@dataclass
class MoleculeGraphMask:
    """Representing a MoleculeGraph with some nodes masked (for GNN training/prediction) and others hidden
    """    
    graph: MoleculeGraph
    sample: DatasetSample
    masked_nodes: np.ndarray
    hidden_nodes: Optional[np.ndarray] = None

@dataclass
class DatasetSampleMask:
    sample: DatasetSample
    masked: Dict[str, pd.Series]
    hidden: Optional[Dict[str, pd.Series]] = None

    def to_graph_mask(self, graph: MoleculeGraph)->MoleculeGraphMask:
        mask_nodes = []
        for molecule, mask in self.masked.items():
            node_mapping = graph.node_mapping[molecule]
            mask_nodes.append(node_mapping.loc[mask, "node_id"].to_numpy())  # type: ignore
        mask_nodes = np.concatenate(mask_nodes)
        hidden_nodes = None
        if self.hidden is not None:
            hidden_nodes = []
            for molecule, mask in self.hidden.items():
                node_mapping = graph.node_mapping[molecule]
                hidden_nodes.append(node_mapping.loc[mask, "node_id"].to_numpy())  # type: ignore
            hidden_nodes = np.concatenate(hidden_nodes)
        return MoleculeGraphMask(graph=graph, sample=self.sample, masked_nodes=mask_nodes, hidden_nodes=hidden_nodes)