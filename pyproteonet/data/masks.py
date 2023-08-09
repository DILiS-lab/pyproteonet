from dataclasses import dataclass
from typing import Optional

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
    molecule: str
    masked: pd.Series
    hidden: Optional[pd.Series] = None

    def to_graph_mask(self, graph: MoleculeGraph)->MoleculeGraphMask:
        node_mapping = graph.node_mapping[self.molecule]
        mask_nodes = node_mapping.loc[self.masked, "node_id"].to_numpy()  # type: ignore
        hidden_nodes = None
        if self.hidden is not None:
            hidden_nodes = node_mapping.loc[self.hidden, "node_id"].to_numpy()  # type: ignore
        return MoleculeGraphMask(graph=graph, sample=self.sample, masked_nodes=mask_nodes, hidden_nodes=hidden_nodes)