from typing import Dict, Optional, List, Iterable, Union
from abc import abstractmethod, ABC

import pandas as pd

from ..dgl.masked_dataset_adapter import MaskedDatasetAdapter


class AbstractMaskedDataset(ABC):

    @property
    @abstractmethod
    def has_hidden(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get_graph_dataset_dgl(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        missing_column_value: Optional[float] = None
    )->MaskedDatasetAdapter:
        raise NotImplementedError()
