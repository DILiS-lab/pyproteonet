from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..data.dataset import Dataset

class Standardizer:
    """
    A class for standardizing and unstandardizing datasets.

    Args:
        columns (List[str]): The columns to be standardized.
        logarithmize (bool, optional): Whether to apply logarithm transformation before standardization. Defaults to False.

    Attributes:
        columns (List[str]): The columns to be standardized.
        logarithmize (bool): Whether to apply logarithm transformation before standardization.
        means (Dict[str, Dict[str, float]]): The means of the standardized columns for each molecule.
        stds (Dict[str, Dict[str, float]]): The standard deviations of the standardized columns for each molecule.
    """

    def __init__(self, columns: List[str], logarithmize: bool = False):
        self.columns = columns
        self.logarithmize = logarithmize
        self.means = {}
        self.stds = {}

    def standardize(self, dataset: "Dataset", inplace: bool = False) -> "Dataset":
        """
        Standardizes the dataset. Standardization parameters are stored in the means and stds attributes to allow for unstandardization.

        Args:
            dataset (Dataset): The dataset to be standardized.
            inplace (bool, optional): Whether to modify the dataset in-place. Defaults to False.

        Returns:
            Dataset: The standardized dataset.
        """
        if not inplace:
            dataset = dataset.copy()
        for mol in dataset.molecules.keys():
            col_means, col_stds = {}, {}
            df = dataset.values[mol].df
            for c in self.columns:
                if c not in df.columns:
                    continue
                vals = df[c]
                if self.logarithmize:
                    vals = np.log(vals)
                mean, std = vals.mean(), vals.std()
                dataset.values[mol][c] = (vals - mean) / std
                col_means[c] = mean
                col_stds[c] = std
            self.means[mol] = col_means
            self.stds[mol] = col_stds
        return dataset

    def unstandardize(self, dataset: "Dataset", inplace: bool = False) -> "Dataset":
        """
        Unstandardizes the dataset by undoing the standardization using the parameters determined during the last call of standardize().

        Args:
            dataset (Dataset): The dataset to be unstandardized.
            inplace (bool, optional): Whether to modify the dataset in-place. Defaults to False.

        Returns:
            Dataset: The unstandardized dataset.
        """
        if not inplace:
            dataset = dataset.copy()
        for mol in dataset.molecules.keys():
            col_means, col_stds = self.means[mol], self.stds[mol]
            df = dataset.values[mol].df
            for c in self.columns:
                if c not in df.columns:
                    continue
                vals = df[c]
                vals = (vals * col_stds[c]) + col_means[c]
                if self.logarithmize:
                    vals = np.exp(vals)
                dataset.values[mol][c] = vals
        return dataset