
from __future__ import annotations
from typing import Optional, TYPE_CHECKING, List, Dict, Hashable

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as snb

from .graph_creation import create_graph_dgl, populate_graph_dgl, NODE_TYPE_INVERSE_MAPPING
if TYPE_CHECKING:
    from .molecule_set import MoleculeSet


class MoleculeGraph:

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, node_mapping: Dict[str, pd.DataFrame],
                  molecule_set: Optional[MoleculeSet], type_mapping: Dict[str, int]):
        self.nodes = nodes
        self.edges = edges
        self.node_mapping = node_mapping
        self.molecule_set = molecule_set
        self.type_mapping = type_mapping
        self.inverse_type_mapping =  {v:k for k,v in type_mapping.items()}

    def to_dgl(self):
        return create_graph_dgl(nodes=self.nodes, edges=self.edges)

    def get_node_degrees(self, molecule_type:str, outgoing:bool=True, incoming:bool=False)->pd.Series:
        edges = []
        if outgoing:
            mask = self.edges.source_node.isin(self.node_mapping[molecule_type].node_id)
            es = self.edges[mask].source_node
            edges.append(es)
        if incoming:
            mask = self.edges.destination_node.isin(self.node_mapping[molecule_type].node_id)
            es = self.edges[mask].destination_node
            edges.append(es)
        edges = pd.concat(edges, ignore_index=True)
        res = pd.Series(data=0, index=self.node_mapping[molecule_type].node_id)
        degs = edges.groupby(edges).count()
        res.loc[degs.index] = degs
        return res

    def plot_node_degrees(self, molecule_type:str, outgoing:bool=True, incoming:bool=False, bins='auto', ax=None):
        deg = self.get_node_degrees(molecule_type=molecule_type, outgoing=outgoing, incoming=incoming)
        if ax is None:
            fig, ax = plt.subplots()
        if bins=='all':
            bins = np.arange(deg.max()+1) + 0.5
        snb.histplot(deg, bins=bins, ax=ax)
        ax.set_xlabel('node degree')
        ax.set_title(f'{molecule_type} node degree')

