from typing import Dict, Literal, TYPE_CHECKING, Callable, Optional, Iterable, List, Union

import pandas as pd
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt
from pathlib import Path
from pandas import HDFStore

from ..dgl.graph_creation import populate_graph_dgl
from .molecule_set import MoleculeSet
from .molecule_graph import MoleculeGraph
from ..utils.numpy import eq_nan

if TYPE_CHECKING:
    from .dataset import Dataset


class DatasetSample:
    def __init__(self, dataset: "Dataset", values: Dict[str, pd.DataFrame], name: str):
        self.dataset = dataset
        self.values = values
        self.name = name
        # self._node_mapping = None
        # self._nodes = None
        # self._edges = None

    def get_index_for(self, molecule_type: Literal["peptide", "protein", "mRNA"]):
        return self.molecules[molecule_type].index

    # def create_graph_nodes_edges(self):
    #     return self.molecule_set.create_graph_nodes_edges()

    def copy(self, columns: Optional[Union[Iterable[str],Dict[str, Union[str, Iterable[str]]]]] = None, molecule_ids: Dict[str, pd.Index] = {}):
        new_values = {}
        for molecule, df in self.values.items():
            if isinstance(columns, dict):
                cs = columns.get(molecule, [])
            else:
                cs = columns
            if cs is None:
                cs = df.keys()
            else:
                if isinstance(cs, str):
                    cs = [cs]
                cs = [c for c in cs if c in df.keys()]
            df = df.loc[:, list(cs)]
            if molecule in molecule_ids:
                df = df.loc[df.index.isin(molecule_ids[molecule])]
            new_values[molecule] = df.copy()
        return DatasetSample(dataset=self.dataset, values=new_values, name=self.name)

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
    def missing_value(self):
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
        target_column: str = 'abundance',
        missing_column_value: Optional[float] = None,
    ):
        populate_graph_dgl(
            graph=self.molecule_set.create_graph(mapping=mapping, bidirectional=True),
            dgl_graph=dgl_graph,
            dataset_sample=self,
            feature_columns=value_columns,
            molecule_columns=molecule_columns,
            target_column=target_column,
            missing_column_value=missing_column_value
        )

    def get_values(self, molecule: str, column: str = 'abundance', return_missing_mask: bool = False):
        values = self.values[molecule][column].to_numpy()
        if return_missing_mask:
            return values, self.missing_mask(molecule=molecule, column=column)
        else:
            return values

    def plot_hist(self, bins="auto"):
        keys = list(self.values.keys())
        fig, ax = plt.subplots(1, len(keys))
        for i, key in enumerate(keys):
            missing_percent = (
                eq_nan(self.values[key].abundance, self.missing_value).sum()
                / self.values[key].abundance.shape[0]
            )
            missing_percent *= 100
            sbn.histplot(self.values[key].abundance, ax=ax[i], bins=bins)
            ax[i].set_title(f"{key} ({round(missing_percent, 1)}% missing)")
        fig.tight_layout()
