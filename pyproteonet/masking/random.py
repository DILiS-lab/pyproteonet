from functools import partial

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset
from .masked_dataset_generator import MaskedDatasetGenerator

def mask_molecule_values_random_non_missing(
    dataset: Dataset,
    molecule: str,
    column: str,
    masking_fraction: float,
) -> MaskedDataset:
    vals = dataset.values[molecule][column]
    vals = vals[~vals.isna()]
    mask_ids = vals.sample(frac=masking_fraction).index
    return MaskedDataset.from_ids(dataset, mask_ids={molecule: mask_ids})

def random_non_missing_generator(
    dataset: Dataset,
    molecule: str,
    column: str,
    masking_fraction: float,
) -> MaskedDatasetGenerator:
    mask_fn = partial(
        mask_molecule_values_random_non_missing,
        molecule=molecule,
        column=column,
        masking_fraction=masking_fraction,
    )
    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask_fn)