from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.dataset import Dataset


def sum_normalize(
    dataset: Dataset,
    molecule: str,
    column: str,
    reference_ids: Optional[pd.Index] = None,
    reference_sample=None,
):
    if reference_sample is None:
        reference_sample = list(dataset.sample_names)[0]
    values = dataset.get_column_flat(molecule=molecule, column=column, ids=reference_ids)
    factors = values.groupby("sample").sum()
    factors = factors[reference_sample] / factors
    values = dataset.get_column_flat(molecule=molecule, column=column)
    factors = values.index.get_level_values("sample").map(factors)
    values = values * factors
    return values


class DnnNormalizer:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.means = {}
        self.stds = {}

    def normalize(self, dataset: Dataset, inplace: bool = False) -> Dataset:
        if not inplace:
            dataset = dataset.copy()
        for mol in dataset.molecules.keys():
            col_means, col_stds = {}, {}
            df = dataset.values[mol].df
            for c in self.columns:
                if c not in df.columns:
                    continue
                vals = np.log(df[c])
                mean, std = vals.mean(), vals.std()
                dataset.values[mol][c] = (vals - mean) / std
                col_means[c] = mean
                col_stds[c] = std
            self.means[mol] = col_means
            self.stds[mol] = col_stds
        return dataset

    def unnormalize(self, dataset: Dataset, inplace: bool = False) -> Dataset:
        if not inplace:
            dataset = dataset.copy()
        for mol in dataset.molecules.keys():
            col_means, col_stds = self.means[mol], self.stds[mol]
            df = dataset.values[mol].df
            for c in self.columns:
                if c not in df.colums:
                    continue
                vals = df[c]
                vals = np.exp((vals * col_stds[c]) + col_means[c])
                dataset.values[mol][c] = vals
        return dataset
