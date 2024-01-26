from typing import Callable, List
import random

from torch.utils.data import IterableDataset

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset


class MaskedDatasetGenerator(IterableDataset):

    def __init__(self, datasets: List[Dataset], generator_fn: Callable[[Dataset], MaskedDataset], sample_wise: bool = False, epoch_size_multiplier: int = 1, shuffle_samplewise_samples: bool = False):
        self.datasets = datasets
        self.generator_fn = generator_fn
        self.sample_wise = sample_wise
        self.epoch_size_multiplier = epoch_size_multiplier
        self.shuffle_samplewise_samples = shuffle_samplewise_samples

    def __len__(self):
        return self.epoch_size_multiplier * (sum([d.num_samples for d in self.datasets]) if self.sample_wise else len(self.datasets))

    def __iter__(self):
        for i in range(self.epoch_size_multiplier):
            for dataset in self.datasets:
                masked_dataset = self.generator_fn(dataset)
                if self.sample_wise:
                    samples = dataset.sample_names
                    if self.shuffle_samplewise_samples:
                        random.shuffle(samples)
                    for sample in samples:
                        yield masked_dataset, [sample]
                else:
                    yield self.generator_fn(dataset), dataset.sample_names