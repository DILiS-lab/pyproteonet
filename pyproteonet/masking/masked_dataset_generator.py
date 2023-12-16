from typing import Callable, List

from torch.utils.data import IterableDataset

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset


class MaskedDatasetGenerator(IterableDataset):

    def __init__(self, datasets: List[Dataset], generator_fn: Callable[[Dataset], MaskedDataset], sample_wise: bool = False):
        self.datasets = datasets
        self.generator_fn = generator_fn
        self.sample_wise = sample_wise

    def __len__(self):
        return sum([d.num_samples for d in self.datasets]) if self.sample_wise else len(self.datasets)

    def __iter__(self):
        for dataset in self.datasets:
            masked_dataset = self.generator_fn(dataset)
            if self.sample_wise:
                for sample in dataset.sample_names:
                    yield masked_dataset, [sample]
            else:
                yield self.generator_fn(dataset), dataset.sample_names