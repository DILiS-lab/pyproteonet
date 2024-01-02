from typing import List, Optional, Iterable, TYPE_CHECKING, Union, Dict, Tuple

import dgl
import torch
import pandas as pd
import numpy as np
from dgl.data import DGLDataset

from ..data.dataset_sample import DatasetSample
from ..data.molecule_set import MoleculeSet

if TYPE_CHECKING:
    from ..data.abstract_masked_dataset import MaskedKeyDataset


class MaskedDatasetAdapter:
    def __init__(
        self,
        masked_dataset: "MaskedKeyDataset",
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        bidirectional_graph: bool = True,
        missing_column_value: Optional[float] = None,
    ):
        if "mask" in value_columns or "mask" in molecule_columns:
            raise KeyError('"mask" is a reserved column name and cannot be part of the value/molecule columns')
        if masked_dataset.has_hidden:
            if "hidden" in value_columns or "hidden" in molecule_columns:
                raise KeyError('"hidden" is a reserved column name and cannot be part of the value/molecule columns')
        self.mapping = mapping
        self.value_columns = value_columns
        self.molecule_columns = molecule_columns
        self.target_column = target_column
        self.masked_dataset = masked_dataset
        self.missing_column_value = missing_column_value
        ms: MoleculeSet = masked_dataset.dataset.molecule_set
        self.graph = ms.create_graph(mapping=mapping, bidirectional=bidirectional_graph)
        self.dgl_graph = self.graph.to_dgl()

    def populate_and_mask(self, sample: DatasetSample, masked_nodes: np.ndarray, hidden_nodes: np.ndarray):
        graph = self.dgl_graph
        sample.populate_graph_dgl(
            dgl_graph=graph,
            mapping=self.mapping,
            value_columns=self.value_columns,
            molecule_columns=self.molecule_columns,
            target_column=self.target_column,
            missing_column_value=self.missing_column_value,
        )
        mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        mask[masked_nodes] = 1
        graph.nodes["molecule"].data["mask"] = mask
        if hidden_nodes is not None and len(hidden_nodes) > 0:
            hide_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
            hide_mask[hidden_nodes] = 1
            graph.nodes["molecule"].data["hidden"] = hide_mask
        return graph
