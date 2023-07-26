from typing import Dict, Literal, TYPE_CHECKING, Callable, Optional, Iterable, List, Union

import pandas as pd
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
from pathlib import Path
from pandas import HDFStore

from .graph_creation import populate_graph_dgl
from .molecule_set import MoleculeSet
from .molecule_graph import MoleculeGraph
from ..utils.numpy import eq_nan

if TYPE_CHECKING:
    from .dataset import Dataset


class DatasetSample:
    def __init__(self, dataset: "Dataset", values: Dict[str, pd.DataFrame]):
        self.dataset = dataset
        self.values = values
        # self._node_mapping = None
        # self._nodes = None
        # self._edges = None

    def get_index_for(self, molecule_type: Literal["peptide", "protein", "mRNA"]):
        return self.molecules[molecule_type].index

    # def create_graph_nodes_edges(self):
    #     return self.molecule_set.create_graph_nodes_edges()

    def copy(self, columns: Optional[Iterable[str]] = None):
        new_values = {}
        for molecule, df in self.values.items():
            cs = columns
            if cs is None:
                cs = df.keys()
            else:
                cs = [c for c in cs if c in df.keys()]
            new_values[molecule] = df.loc[:, list(cs)]
        return DatasetSample(dataset=self.dataset, values=new_values)

    def missing_mask(self, molecule: str, column: str = "abundance"):
        return eq_nan(self.values[molecule].loc[:, column], self.dataset.missing_value)

    def non_missing_mask(self, molecule: str, column: str = "abundance"):
        return ~self.missing_mask(molecule=molecule, column=column)

    def missing_molecules(self, molecule: str, column: str = "abundance"):
        mask = self.missing_mask(molecule=molecule, column=column)
        return self.molecules[molecule].loc[self.values[molecule][mask].index, :]

    def non_missing_molecules(self, molecule: str, column: str = "abundance"):
        mask = self.non_missing_mask(molecule=molecule, column=column)
        return self.molecules[molecule].loc[self.values[molecule][mask].index, :]

    def apply(self, fn: Callable, *args, **kwargs):
        return fn(self, *args, **kwargs)

    @property
    def molecule_set(self) -> MoleculeSet:
        return self.dataset.molecule_set

    @property
    def missing_abundance_value(self):
        return self.dataset.missing_value

    @property
    def missing_label_value(self):
        return self.dataset.missing_label_value

    @property
    def gene_mapping(self):
        return self.molecule_set.mappings

    @property
    def molecules(self):
        return self.molecule_set.molecules

    # @property
    # def nodes(self):
    #     return self.molecule_set.nodes

    # @property
    # def edges(self):
    #     return self.molecule_set.edges

    # @property
    # def node_mapping(self):
    #     return self.molecule_set.node_mapping

    def get_node_values_for_graph(self, graph: MoleculeGraph, include_id_and_type: bool = True):
        node_values = []
        for node_type, df in graph.nodes.groupby("type"):
            key = graph.inverse_type_mapping[node_type]  # type: ignore
            values = self.values[key]
            columns = list(values.columns)
            df.loc[:, columns] = self.dataset.missing_value
            mask = df.molecule_id.isin(values.index)
            df.loc[mask, columns] = values.loc[df.loc[mask, "molecule_id"], columns].to_numpy()
            if include_id_and_type:
                node_values.append(df)
            else:
                node_values.append(df.loc[:, columns])
        node_values = pd.concat(node_values)
        return node_values

    # def create_graph_dgl(self):
    #     graph = self.molecule_set.create_graph_dgl()
    #     self.populate_graph_dgl(graph)
    #     return graph

    def populate_graph_dgl(
        self,
        dgl_graph,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = 'abundance'
    ):
        populate_graph_dgl(
            graph=self.molecule_set.create_graph(mapping=mapping, bidirectional=True),
            dgl_graph=dgl_graph,
            dataset_sample=self,
            value_columns=value_columns,
            molecule_columns=molecule_columns,
            target_column=target_column
        )

    def plot_hist(self, bins="auto"):
        keys = list(self.values.keys())
        fig, ax = plt.subplots(1, len(keys))
        for i, key in enumerate(keys):
            missing_percent = (
                eq_nan(self.values[key].abundance, self.missing_abundance_value).sum()
                / self.values[key].abundance.shape[0]
            )
            missing_percent *= 100
            sbn.histplot(self.values[key].abundance, ax=ax[i], bins=bins)
            ax[i].set_title(f"{key} ({round(missing_percent, 1)}% missing)")
        fig.tight_layout()
