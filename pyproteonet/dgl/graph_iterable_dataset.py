from typing import TYPE_CHECKING

from torch.utils.data import IterableDataset

from ..data.dataset_sample import DatasetSample
from ..data.masks import DatasetSampleMask
from .masked_dataset_adapter import MaskedDatasetAdapter

if TYPE_CHECKING:
    from ..data.masked_dataset_iterable import MaskedDatasetIterable


class GraphIterableDataset(MaskedDatasetAdapter, IterableDataset):
    def __init__(self, masked_dataset: "MaskedDatasetIterable", **kwds):
        super().__init__(masked_dataset=masked_dataset, **kwds)

    def __len__(self):
        return len(self.masked_dataset)

    def __iter__(self):
        for sample_mask in self.masked_dataset:
            graph_mask = sample_mask.to_graph_mask(graph=self.graph)
            yield self.populate_and_mask(
                sample=sample_mask.sample, masked_nodes=graph_mask.masked_nodes, hidden_nodes=graph_mask.hidden_nodes
            )
