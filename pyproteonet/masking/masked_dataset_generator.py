from typing import Callable, List

from torch.utils.data import IterableDataset

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset


class MaskedDatasetGenerator(IterableDataset):

    def __init__(self, datasets: List[Dataset], generator_fn: Callable[[Dataset], MaskedDataset], sample_wise: bool = False, epoch_size_multiplier: int = 1):
        self.datasets = datasets
        self.generator_fn = generator_fn
        self.sample_wise = sample_wise
        self.epoch_size_multiplier = epoch_size_multiplier

    def __len__(self):
        return self.epoch_size_multiplier * (sum([d.num_samples for d in self.datasets]) if self.sample_wise else len(self.datasets))

    def __iter__(self):
        for i in range(self.epoch_size_multiplier):
            for dataset in self.datasets:
                masked_dataset = self.generator_fn(dataset)
                if self.sample_wise:
                    for sample in dataset.sample_names:
                        yield masked_dataset, [sample]
                else:
                    yield self.generator_fn(dataset), dataset.sample_names