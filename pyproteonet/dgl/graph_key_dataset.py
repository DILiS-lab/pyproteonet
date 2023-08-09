from typing import TYPE_CHECKING

import torch

from ..data.dataset_sample import DatasetSample
from .masked_dataset_adapter import MaskedDatasetAdapter

if TYPE_CHECKING:
    from ..data.masked_dataset import MaskedDataset


class GraphKeyDataset(MaskedDatasetAdapter, torch.utils.data.Dataset):
    def __init__(self, masked_dataset: "MaskedDataset", **kwds):
        super().__init__(masked_dataset=masked_dataset, **kwds)
        self.keys = list(masked_dataset.keys())
        self.masked_dataset: "MaskedDataset"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        masked_nodes = self.masked_dataset.get_masked_nodes(key, graph=self.graph)
        hidden_nodes = None
        if self.masked_dataset.has_hidden:
            hidden_nodes = self.masked_dataset.get_hidden_nodes(key=key, graph=self.graph)
        sample = self.masked_dataset.get_sample(key)
        return self.populate_and_mask(sample=sample, masked_nodes=masked_nodes, hidden_nodes=hidden_nodes)

    def index_to_sample(self, i: int) -> DatasetSample:
        return self.masked_dataset.get_sample(self.keys[i])
