from typing import (
    Dict,
    Literal,
    TYPE_CHECKING,
    Callable,
    Optional,
    Iterable,
    List,
    Union,
)

import pandas as pd
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt

from .molecule_set import MoleculeSet
from .molecule_graph import MoleculeGraph
from ..utils.numpy import eq_nan

if TYPE_CHECKING:
    from .dataset import Dataset


class DatasetSample:
    """
    Representing a dataset samples holding a set of values for every molecule. Can be thought of as a dictionary of pandas dataframes with one dataframe for each molecule.
    """

    def __init__(self, dataset: "Dataset", values: Dict[str, pd.DataFrame], name: str):
        """Create a dataset samples holding a set of values for every molecule

        Args:
            dataset (Dataset): The dataset this sample belongs to.
            values (Dict[str, pd.DataFrame]): Values for every molecule in the dataset.
            name (str): Name of the sample.
        """
        self.dataset = dataset
        self.values = values
        self.name = name

    def get_index_for(self, molecule_type: str) -> pd.Index:
        """returns the index of molecule ids for the given molecule type

        Args:
            molecule_type (str): The molecule type to get the index for

        Returns:
            pd.Index: The index of molecule ids for the given molecule type
        """
        return self.molecules[molecule_type].index

    def copy(
        self,
        columns: Optional[
            Union[Iterable[str], Dict[str, Union[str, Iterable[str]]]]
        ] = None,
        molecule_ids: Dict[str, pd.Index] = {},
    )->"DatasetSample":
        """Creates a copy of the dataset sample.

        Args:
            columns (Optional[ Union[Iterable[str], Dict[str, Union[str, Iterable[str]]]] ], optional): Columns to copy.
              When given as list of strings the same columns are copied for every molecule, when given as dictionary the key specific
              columns can be specific per molecule type. Defaults to None.
            molecule_ids (Dict[str, pd.Index], optional): Dictionay specifying for every molecule type the molecule ids that will be copied.
              If a molecule type is not part of the dictionary all molecule ids will be copied for this molecule type. Defaults to {}.

        Returns:
            DatasetSamples: A copy of the dataset sample.
        """
        new_values = {}
        for molecule, df in self.values.items():
            if isinstance(columns, dict):
                cs = columns.get(molecule, [])
            else:
                cs = columns
            if cs is None:
                cs = df.keys()
            else:
                if isinstance(cs, str):
                    cs = [cs]
                cs = [c for c in cs if c in df.keys()]
            df = df.loc[:, list(cs)]
            if molecule in molecule_ids:
                df = df.loc[df.index.isin(molecule_ids[molecule])]
            new_values[molecule] = df.copy()
        return DatasetSample(dataset=self.dataset, values=new_values, name=self.name)

    def missing_mask(self, molecule: str, column: str = "abundance")->np.ndarray:
        """Returns a boolean mask indicating which values are missing for the given molecule and column.

        Args:
            molecule (str): the molecule type (e.g. 'protein' or 'peptide')
            column (str, optional): the value column. Defaults to "abundance".

        Returns:
            np.ndarray: the boolean mask indicating which values are missing for the given molecule and column.
        """
        return eq_nan(self.values[molecule].loc[:, column], self.dataset.missing_value)

    def non_missing_mask(self, molecule: str, column: str = "abundance"):
        """Returns a boolean mask indicating which values are non-missing for the given molecule and column.

        Args:
            molecule (str): the molecule type (e.g. 'protein' or 'peptide')
            column (str, optional): the value column. Defaults to "abundance".

        Returns:
            np.ndarray: the boolean mask indicating which values are non-missing for the given molecule and column.
        """
        return ~self.missing_mask(molecule=molecule, column=column)

    def missing_molecules(self, molecule: str, column: str = "abundance")->pd.DataFrame:
        """Returns all molecules of the given molecule type that are missing for the given column.

        Args:
            molecule (str): the molecule type (e.g. 'protein' or 'peptide')
            column (str, optional): the value column. Defaults to "abundance".

        Returns:
            pd.DataFrame: the dataframe containing the missing molecules and their additional information for the given molecule and column.
        """
        mask = self.missing_mask(molecule=molecule, column=column)
        return self.molecules[molecule].loc[self.values[molecule][mask].index, :]

    def non_missing_molecules(self, molecule: str, column: str = "abundance"):
        """Returns all molecules of the given molecule type that are not missing for the given column.

        Args:
            molecule (str): the molecule type (e.g. 'protein' or 'peptide')
            column (str, optional): the value column. Defaults to "abundance".

        Returns:
            pd.DataFrame: the dataframe containing the non-missing molecules and their additional information for the given molecule and column.
        """
        mask = self.non_missing_mask(molecule=molecule, column=column)
        return self.molecules[molecule].loc[self.values[molecule][mask].index, :]

    def apply(self, fn: Callable, *args, **kwargs)->object:
        """Applies a function to the dataset sample. Only exists to match the interface of the Dataset class.

        Args:
            fn (Callable): the function to apply

        Returns:
            object: the result of the function
        """
        return fn(self, *args, **kwargs)

    @property
    def molecule_set(self) -> MoleculeSet:
        return self.dataset.molecule_set

    @property
    def missing_value(self):
        return self.dataset.missing_value

    @property
    def missing_label_value(self):
        return self.dataset.missing_label_value

    @property
    def gene_mapping(self):
        return self.molecule_set.mappings

    @property
    def molecules(self):
        return self.molecule_set.molecules

    def get_node_values_for_graph(
        self, graph: MoleculeGraph, include_id_and_type: bool = True
    )->pd.DataFrame:
        """Returns the values for the given graph.

        Args:
            graph (MoleculeGraph): the graph to get the values for
            include_id_and_type (bool, optional): Whether to include the molecule ids and molecule type into the result. Defaults to True.

        Returns:
            pd.DataFrame: the values for the given graph
        """
        node_values = []
        for node_type, df in graph.nodes.groupby("type"):
            key = graph.inverse_type_mapping[node_type]  # type: ignore
            values = self.values[key]
            columns = list(values.columns)
            df.loc[:, columns] = self.dataset.missing_value
            mask = df.molecule_id.isin(values.index)
            df.loc[mask, columns] = values.loc[
                df.loc[mask, "molecule_id"], columns
            ].to_numpy()
            if include_id_and_type:
                node_values.append(df)
            else:
                node_values.append(df.loc[:, columns])
        node_values = pd.concat(node_values)
        return node_values

    def plot_hist(self, bins: Union[List[float], str]="auto"):
        """Plots a histogram of the values for every molecule type.

        Args:
            bins (str, optional): The bins for the histogram (passed to seaborn.histplot). Defaults to "auto".
        """
        keys = list(self.values.keys())
        fig, ax = plt.subplots(1, len(keys))
        for i, key in enumerate(keys):
            missing_percent = (
                eq_nan(self.values[key].abundance, self.missing_value).sum()
                / self.values[key].abundance.shape[0]
            )
            missing_percent *= 100
            sbn.histplot(self.values[key].abundance, ax=ax[i], bins=bins)
            ax[i].set_title(f"{key} ({round(missing_percent, 1)}% missing)")
        fig.tight_layout()
