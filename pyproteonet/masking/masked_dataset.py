from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import dgl
import torch

from ..data.dataset_sample import DatasetSample
from ..data.molecule_graph import MoleculeGraph
from ..data.dataset import Dataset

def _ids_to_mask(dataset: Dataset, molecule: str, ids: pd.Index):
    mask = pd.DataFrame(
        index=dataset.molecules[molecule].index,
        data={sample: False for sample in dataset.sample_names},
    )
    if "sample" in ids.names:
        m = pd.Series(index=ids, data=True)
        m = m.unstack(level="sample", fill_value=False)
        mask.loc[m.index, m.columns] = m
    else:
        for sample in dataset.sample_names:
            mask.loc[ids, sample] = True
    return mask

class MaskedDataset():
    """A dataset with some molecules masked. Used for self supervised training of predictive imputation models.
    
    Attributes:
        dataset (Dataset): The original dataset.
        masks (Dict[str, pd.DataFrame]): A dictionary mapping molecule names to boolean DataFrames reprenting the masks. Defaults to an empty dictionary.
        hidden (Optional[Dict[str, pd.DataFrame]]): A dictionary mapping molecule names to boolean DataFrames reprenting hidden molecules (optional). Defaults to None.
    """
    def __init__(
        self,
        dataset: Dataset,
        masks: Dict[str, pd.DataFrame] = {},
        hidden: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.dataset = dataset
        self._keys = set.union(*[set(m.keys()) for m in masks.values()])
        self.masks = masks
        self.hidden = dict()
        if hidden is not None:
            self.hidden = hidden
            self._keys = set.union(
                self._keys, *[set(m.keys()) for m in hidden.values()]
            )
        self._keys = list(self._keys)

    @classmethod
    def from_ids(
        cls,
        dataset: Dataset,
        mask_ids: Dict[str, pd.Index],
        hidden_ids: Optional[Dict[str, pd.Index]] = None,
    ) -> "MaskedDataset":
        """
        Create a MaskedDataset object from a given dataset and mask IDs.

        Args:
            dataset (Dataset): The original dataset.
            mask_ids (Dict[str, pd.Index]): A dictionary giving for each molecule type a list of molecule ids to mask.
            hidden_ids (Optional[Dict[str, pd.Index]]): A dictionary giving for each molecule type a list of molecule ids to mask. (default: None).

        Returns:
            MaskedDataset: A new MaskedDataset object with the specified masked and hidden molecules.
        """
        masks = dict()
        for mol, ids in mask_ids.items():
            masks[mol] = _ids_to_mask(dataset=dataset, molecule=mol, ids=ids)
        hidden = None
        if hidden_ids is not None:
            hidden = dict()
            for mol, ids in hidden_ids.items():
                hidden[mol] = _ids_to_mask(dataset=dataset, molecule=mol, ids=ids)
        return cls(dataset=dataset, masks=masks, hidden=hidden)

    def set_mask(self, molecule: str, mask: pd.DataFrame) -> None:
        """
        Set the mask for a specific molecule type.

        Args:
            molecule (str): The name of the molecule type.
            mask (pd.DataFrame): The mask dataframe.

        Returns:
            None
        """
        self.masks[molecule] = mask

    def get_mask_ids(self, molecule: str) -> pd.Index:
            """
            For a given molecule type get the ids of the masked molecules.

            Args:
                molecule (str): The molecule type for which to retrieve the masked IDs.

            Returns:
                pd.Index: The masked IDs for the given molecule type.
            """
            ids = self.masks[molecule].stack().swaplevel()
            ids = ids[ids]
            return ids.index.set_names(["sample", "id"])

    def set_mask_ids(self, molecule: str, ids: pd.Index) -> None:
        """
        For a specific molecule type sets the masked molecules (given by their ids).

        Args:
            molecule (str): The name of the molecule type.
            ids (pd.Index): The IDs to be masked.

        Returns:
            None
        """
        self.masks[molecule] = _ids_to_mask(
            dataset=self.dataset, molecule=molecule, ids=ids
        )

    def set_hidden(self, molecule: str, hidden: pd.DataFrame) -> None:
        """
        Sets the hidden-mask (which molecules to hide) for a specific molecule type.

        Args:
            molecule (str): The name of the molecule type.
            mask (pd.DataFrame): The mask dataframe.

        Returns:
            None
        """
        self.hidden[molecule] = hidden

    def get_hidden_ids(self, molecule: str) -> pd.Index:
        """
        For a specific molecule type sets the masked molecules (given by their ids).

        Args:
            molecule (str): The name of the molecule type.
            ids (pd.Index): The IDs to be masked.

        Returns:
            None
        """
        ids = self.hidden[molecule].stack().swaplevel()
        ids = ids[ids]
        return ids.index.set_names(["sample", "id"])

    def set_hidden_ids(self, molecule: str, ids: pd.Index) -> None:
        """
        For a specific molecule type sets the hidden molecules (given by their ids).

        Args:
            molecule (str): The name of the molecule type.
            ids (pd.Index): The IDs to be hidden.

        Returns:
            None
        """
        self.hidden[molecule] = _ids_to_mask(
            dataset=self.dataset, molecule=molecule, ids=ids
        )

    def keys(self) -> Iterable[str]:
        """
        Names of samples that have eithe masked or hidden molecules.

        Returns:
            Iterable[str]: An iterable of keys in the dataset.
        """
        return self._keys

    @property
    def has_hidden(self) -> bool:
        """
        Check if the dataset has any hidden molecules.

        Returns:
            bool: True if the dataset has hidden molecules, False otherwise.
        """
        if len(self.hidden):
            return True
        return False

    def get_sample(self, key: str) -> DatasetSample:
        """
        Retrieves a dataset sample by its key.

        Args:
            key (str): The key of the sample to retrieve.

        Returns:
            DatasetSample: The dataset sample corresponding to the given key.
        """
        return self.dataset[key]

    def set_samples_value_matrix(
        self,
        matrix: Union[np.array, torch.tensor, pd.DataFrame],
        molecule: str,
        column: str,
        samples: Optional[List[str]] = None,
        only_set_masked: bool = True,
    ) -> None:
        """
        Sets the values of a value column of the underlying to those by by a matrix (numpy array, torch tensor, or pandas DataFrame).
        If specified only values for the masked molecules are set. Useful to write back the results of a model imputation.

        Args:
            matrix (Union[np.array, torch.tensor, pd.DataFrame]): The value matrix to be set.
            molecule (str): The name of the molecule type (e.g. protein, peptide...).
            column (str): The name of the value column.
            samples (Optional[List[str]], optional): The list of names of samples to consider. Defaults to None.
            only_set_masked (bool, optional): If True, only sets the values for masked molecules. 
                If False, sets the values for all molecules. Defaults to True.

        Raises:
            ValueError: If the provided sample names do not match the column names in the matrix.

        Returns:
            None
        """
        if isinstance(matrix, pd.DataFrame):
            if samples is None:
                samples = matrix.columns
            else:
                if set(samples) != set(matrix.columns):
                    raise ValueError(
                        "If samples names are provided the column names in the matrix must match the samples names"
                    )
            matrix = matrix.values
        elif isinstance(matrix, torch.Tensor):
            matrix = matrix.numpy()
        if samples is None:
            samples = self.dataset.sample_names
        mol_ids = self.dataset.molecules[molecule].index
        mat_df = self.dataset.get_samples_value_matrix(molecule=molecule, column=column, samples=samples)
        mat_df = mat_df.loc[mol_ids]
        if only_set_masked:
            mask = self.masks[molecule].loc[mol_ids, samples].values
            mat_df.mask(mask, matrix, inplace=True)
        else:
            mat_df.values[:, :] = matrix
        self.dataset.set_wf(
            matrix=mat_df, molecule=molecule, column=column
            )

    def to_dgl_graph(
        self,
        feature_columns: Dict[str, Union[str, List[str]]],
        mappings: Union[str, List[str]],
        molecule_columns: Dict[str, Union[str, List[str]]] = {},
        mapping_directions: Dict[str, Tuple[str, str]] = {},
        make_bidirectional: bool = False,
        features_to_float32: bool = True,
        samples: Optional[List[str]] = None
    ) -> dgl.DGLHeteroGraph:
        """
        Converts the masked dataset to a DGL heterograph.

        Args:
            feature_columns (Dict[str, Union[str, List[str]]]): Dictionary specifying the feature columns for each molecule type.
            mappings (Union[str, List[str]]): List of mapping names or a single mapping name to be used for constructing the graph.
            molecule_columns (Dict[str, Union[str, List[str]]], optional): Dictionary specifying the molecule columns for each molecule type. Defaults to {}.
            mapping_directions (Dict[str, Tuple[str, str]], optional): Dictionary specifying the mapping directions (Tuple of molecule types) for each mapping. Defaults to {}.
            make_bidirectional (bool, optional): Whether to make the graph bidirectional by adding reverse edges. Defaults to False.
            features_to_float32 (bool, optional): Whether to convert the features to float32. Defaults to True.
            samples (Optional[List[str]], optional): List of sample names to include in the graph. Defaults to None.

        Returns:
            dgl.DGLHeteroGraph: The converted DGL heterograph.
        """
        g = self.dataset.to_dgl_graph(
            feature_columns=feature_columns,
            mappings=mappings,
            molecule_columns=molecule_columns,
            mapping_directions=mapping_directions,
            make_bidirectional=make_bidirectional,
            features_to_float32=features_to_float32,
            samples=samples
        )
        if samples is None:
            samples = self.dataset.sample_names
        num_samples = len(samples)
        for mol, mol_features in feature_columns.items():
            mol_ids = self.dataset.molecules[mol].index
            if mol in self.masks:
                mask = torch.from_numpy(
                    self.masks[mol].loc[mol_ids, samples].to_numpy()
                )
            else:
                mask = torch.full(
                    (mol_ids.shape[0], num_samples), False
                )
            if mol in self.hidden:
                hidden = torch.from_numpy(
                    self.hidden[mol].loc[mol_ids, samples].to_numpy()
                )
            else:
                hidden = torch.full(
                    (mol_ids.shape[0], num_samples), False
                )
            g.nodes[mol].data["mask"] = mask
            g.nodes[mol].data["hidden"] = hidden
        return g