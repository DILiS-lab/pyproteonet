from typing import Callable, List

from torch.utils.data import IterableDataset

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset


class MaskedDatasetGenerator(IterableDataset):

    def __init__(self, datasets: List[Dataset], generator_fn: Callable[[Dataset], MaskedDataset]):
        self.datasets = datasets
        self.generator_fn = generator_fn

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        for dataset in self.datasets:
            yield self.generator_fn(dataset)