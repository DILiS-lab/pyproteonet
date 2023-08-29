from typing import Dict, Optional, List, Iterable, Union, Callable

import pandas as pd

from .dataset import Dataset
from .dataset_sample import DatasetSample
from .masks import MoleculeGraphMask
from ..dgl.graph_iterable_dataset import GraphIterableDataset
from ..data.masks import DatasetSampleMask


class MaskedDatasetIterable:
    def __init__(
        self,
        dataset: Dataset,
        mask_function: Callable[[DatasetSample], MoleculeGraphMask],
        has_hidden: bool = True,
    ) -> None:
        self.dataset = dataset
        self.mask_function = mask_function
        self._has_hidden = has_hidden

    def keys(self) -> Iterable[str]:
        return self.mask.keys()

    @property
    def has_hidden(self) -> bool:
        return self._has_hidden

    def __len__(self):
        return len(self.dataset.samples)

    def __iter__(self)->Iterable[DatasetSampleMask]:
        for sample in self.dataset.samples:
            yield self.mask_function(sample)

    def get_graph_dataset_dgl(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        missing_column_value: Optional[float] = None
    )->GraphIterableDataset:
        return GraphIterableDataset(
            masked_dataset=self,
            mapping=mapping,
            value_columns=value_columns,
            molecule_columns=molecule_columns,
            target_column=target_column,
            missing_column_value=missing_column_value
        )
