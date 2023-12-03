import pandas as pd
import numpy as np

from typing import Optional, Union

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset
from ..simulation.utils import get_numpy_random_generator


def mask_missing(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
) -> MaskedDataset:
    mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    for sample_name, sample in dataset.samples_dict.items():
        mask[sample_name] = False
        mask.loc[sample.missing_mask(molecule=molecule, column=column), sample_name] = True
    return MaskedDataset(dataset=dataset, masks={molecule: mask})

def mask_non_missing(dataset: Dataset, 
    molecule: str,
    column: str,
    ids: Optional[pd.Index] = None,
    frac: Optional[float] = None,
    random_seed: Optional[Union[int, np.random.Generator]] = None):
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