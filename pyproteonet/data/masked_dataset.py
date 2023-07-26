from typing import Dict, Optional, List, Iterable, Union

import pandas as pd

from .dataset import Dataset
from .dataset_sample import DatasetSample
from .molecule_graph import MoleculeGraph
from ..dgl.graph_data_set import GraphDataSet


class MaskedDataset:
    def __init__(
        self, dataset: Dataset, mask: pd.DataFrame, hidden: Optional[pd.DataFrame] = None, molecule: str = "protein"
    ) -> None:
        self.dataset = dataset
        self.mask = mask
        self.hidden = hidden
        self.molecule = molecule

    def keys(self) -> Iterable[str]:
        return self.mask.keys()

    def has_hidden(self) -> bool:
        if self.hidden is not None:
            return True
        return False

    def get_sample(self, key: str) -> DatasetSample:
        return self.dataset[key]

    def get_masked_nodes(self, key: str, graph: MoleculeGraph) -> Iterable[int]:
        node_mapping = graph.node_mapping[self.molecule]
        mask_nodes = self.mask[key]
        mask_nodes = mask_nodes.loc[mask_nodes].index
        mask_nodes = node_mapping.loc[mask_nodes, "node_id"].to_numpy()  # type: ignore
        return mask_nodes

    def get_hidden_nodes(self, key: str, graph: MoleculeGraph) -> Iterable[int]:
        if self.hidden is None or key not in self.hidden.columns:
            return []
        node_mapping = graph.node_mapping[self.molecule]
        hidden_nodes = self.hidden[key]
        hidden_nodes = hidden_nodes.loc[hidden_nodes].index
        return node_mapping.loc[hidden_nodes, "node_id"].to_numpy()  # type: ignore

    def get_graph_dataset_dgl(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
    ):
        return GraphDataSet(
            masked_datasets=[self], mapping=mapping, value_columns=value_columns, molecule_columns=molecule_columns, target_column=target_column
        )
