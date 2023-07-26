from typing import List, Optional, Iterable, TYPE_CHECKING, Union, Dict

import dgl
import torch
import pandas as pd
import numpy as np
from dgl.data import DGLDataset

from ..data.dataset_sample import DatasetSample
from ..data.molecule_graph import MoleculeGraph

if TYPE_CHECKING:
    from ..data.masked_dataset import MaskedDataset


class GraphDataSet(DGLDataset):
    def __init__(
        self,
        masked_datasets: Iterable["MaskedDataset"],
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        bidirectional_graph: bool = True,
    ):
        if "mask" in value_columns or "mask" in molecule_columns:
            raise KeyError('"mask" is a reserved column name and cannot be part of the value/molecule columns')
        if any([md.has_hidden() for md in masked_datasets]):
            if "hide" in value_columns or "hide" in molecule_columns:
                raise KeyError('"hide" is a reserved column name and cannot be part of the value/molecule columns')
        self.value_columns = value_columns
        self.molecule_columns = molecule_columns
        self.target_column=target_column
        self.dataset_samples: List[DatasetSample] = []
        self.mask_nodes = []
        self.hide_nodes = []
        self.graphs = dict()
        for masked_dataset in masked_datasets:
            ms = masked_dataset.dataset.molecule_set
            graph = ms.create_graph(mapping=mapping, bidirectional=bidirectional_graph)
            #node_mapping = graph.node_mapping[masked_dataset.molecule]
            if ms.id not in self.graphs:
                self.graphs[ms.id] = (graph, graph.to_dgl())
            for mask_name in masked_dataset.keys():
                sample = masked_dataset.get_sample(mask_name)
                self.dataset_samples.append(sample)
                mask_nodes = masked_dataset.get_masked_nodes(mask_name, graph=graph)
                self.mask_nodes.append(mask_nodes)
                hide_mask = masked_dataset.get_hidden_nodes(mask_name, graph=graph)
                self.hide_nodes.append(hide_mask)
        # self.indices = []
        # for i, ms in enumerate(self.mask_nodes):
        #     assert len(ms)
        #     self.indices.extend([(i,ms_index) for ms_index in range(len(ms))])

    def __len__(self):
        return len(self.mask_nodes)

    def __getitem__(self, i):
        # ds_i, ms_i = self.indices[i]
        sample: DatasetSample = self.dataset_samples[i]
        ms = self.mask_nodes[i]
        graph = self.graphs[sample.molecule_set.id][1]
        sample.populate_graph_dgl(graph, value_columns=self.value_columns, molecule_columns=self.molecule_columns, target_column=self.target_column)
        mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        mask[ms] = 1
        graph.nodes["molecule"].data["mask"] = mask
        if len(self.hide_nodes) > 0:
            hide = self.hide_nodes[i]
            hide_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            hide_mask[hide] = 1
            graph.nodes["molecule"].data["hide"] = hide_mask
        return graph

    def index_to_sample(self, i: int)->DatasetSample:
        return self.dataset_samples[i]
    
    def index_to_molecule_graph(self, i: int)->MoleculeGraph:
        sample = self.dataset_samples[i]
        return self.graphs[sample.molecule_set.id][0]

# class GraphDataSetDeprecated(DGLDataset):
#     def __init__(
#         self,
#         dataset_samples: List[DatasetSample] = [],
#         mask_nodes: List[List[np.ndarray]] = [],
#         hide_nodes: Optional[List[np.ndarray]] = None,
#         mapping: str = "gene",
#         value_columns: List[str] = ["abundance"],
#         molecule_columns: List[str] = [],
#     ):
#         assert len(dataset_samples) == len(mask_nodes)
#         if hide_nodes is not None:
#             assert len(dataset_samples) == len(hide_nodes)
#         self.dataset_samples = dataset_samples
#         self.mask_nodes = mask_nodes
#         self.hide_nodes = hide_nodes
#         self.graphs = dict()
#         self.value_columns = value_columns
#         self.molecule_columns = molecule_columns
#         for ds in dataset_samples:
#             if ds.molecule_set.id not in self.graphs:
#                 graph = ds.molecule_set.create_graph(mapping=mapping).to_dgl()
#                 assert "mask" not in ds.values.keys()
#                 if self.mask_nodes is not None:
#                     assert "hide" not in ds.values.keys()
#                 self.graphs[ds.molecule_set.id] = graph
#         self.indices = []
#         for i, ms in enumerate(mask_nodes):
#             assert len(ms)
#             self.indices.extend([(i, ms_index) for ms_index in range(len(ms))])

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, i):
#         ds_i, ms_i = self.indices[i]
#         sample = self.dataset_samples[ds_i]
#         ms = self.mask_nodes[ds_i][ms_i]
#         graph = self.graphs[sample.molecule_set.id]
#         sample.populate_graph_dgl(
#             graph, value_columns=self.value_columns, molecule_columns=self.molecule_columns,
#         )
#         mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
#         mask[ms] = 1
#         graph.nodes["molecule"].data["mask"] = mask
#         if self.hide_nodes is not None:
#             hide = self.hide_nodes[ds_i]
#             hide_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
#             hide_mask[hide] = 1
#             graph.nodes["molecule"].data["hide"] = hide_mask
#         return graph
