import pandas as pd
import numpy as np

from typing import Optional, Union, Dict

from ..data.dataset import Dataset
from .masked_dataset import MaskedDataset
from ..simulation.utils import get_numpy_random_generator


def mask_missing(
    dataset: Dataset,
    molecule_columns: Dict[str, str]
) -> MaskedDataset:
    """
    Creates a MaskedDataset with missing values for a the specified molecules and value columns masked.

    Args:
        dataset (Dataset): The dataset containing the data to be masked.
        molecule_columns (Dict[str, str]): A dictionary mapping molecule names to value columns containing missing values used for masking.

    Returns:
        MaskedDataset: The masked dataset.

    """
    masks = dict()
    for molecule, column in molecule_columns.items():
        mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
        for sample_name, sample in dataset.samples_dict.items():
            mask[sample_name] = False
            mask.loc[sample.missing_mask(molecule=molecule, column=column), sample_name] = True
        masks[molecule] = mask
    return MaskedDataset(dataset=dataset, masks=masks)

def mask_non_missing(dataset: Dataset, 
    molecule: str,
    column: str,
    ids: Optional[pd.Index] = None,
    frac: Optional[float] = None,
    random_seed: Optional[Union[int, np.random.Generator]] = None):
    """
    Masks non-missing values in a dataset for specified molecule and column.

    Args:
        dataset (Dataset): The dataset to mask.
        molecule (str): The molecule to consider.
        column (str): The column to consider.
        ids (Optional[pd.Index], optional): If given only mask molecules with those ids. Defaults to None.
        frac (Optional[float], optional): If given only masks this fraction of all molecule valid for masking. Defaults to None.
        random_seed (Optional[Union[int, np.random.Generator]], optional): The random seed for reproducibility. Defaults to None.

    Returns:
        MaskedDataset: The masked dataset.

    """
    mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    rng = get_numpy_random_generator(seed=random_seed)
    for sample_name, sample in dataset.samples_dict.items():
        mask[sample_name] = False
        masked_mols = sample.non_missing_molecules(molecule=molecule, column=column)
        if ids is not None:
            masked_mols = masked_mols[masked_mols.index.isin(ids)].index
        if frac is not None:
            masked_mols = masked_mols.sample(frac=frac, random_state=rng.integers(np.iinfo(np.int32).max)).index
        mask.loc[masked_mols, sample_name] = True
    return MaskedDataset(dataset=dataset, masks={molecule: mask})