from typing import Callable, List
import random

from torch.utils.data import IterableDataset

from pyproteonet.masking.masked_dataset import MaskedDataset

from ..data.dataset import Dataset


class MaskedDatasetGenerator(IterableDataset):
    """
    Itarable set of MaskedDatasets and sample names. The MaskedDatasets are generated from a given list of Datasets using a given generator function that takes a Dataset and returns a MaskedDataset.
    The sample_names given with each MaskedDataset are either a list of all samples or just a single simple if sample_wise is True.
    This can be used by a DataLoader to generate MaskedDatasets on the fly.


    Args:
        datasets (List[Dataset]): List of datasets to generate masked datasets from.
        generator_fn (Callable[[Dataset], MaskedDataset]): Function that generates a masked dataset from a given dataset.
        sample_wise (bool, optional): If True, returns each MaskedDataset num_samples times each times accompied by another sample name. If false just returns the MaskedDataset together with a list of all sample names. Defaults to False.
        epoch_size_multiplier (int, optional): Multiplier for the number of iterations over the datasets to make up one "epoch". Defaults to 1.
        shuffle_samplewise_samples (bool, optional): If True and sample_wise is True, shuffles the samples within each dataset. Defaults to False.
    """

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