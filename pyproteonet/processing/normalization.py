from typing import List

import numpy as np

from ..data.dataset import Dataset

class DnnNormalizer:
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.means = {}
        self.stds = {}

    def normalize(self, dataset: Dataset, inplace: bool = False)->Dataset:
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

    def unnormalize(self, dataset: Dataset, inplace: bool = False)->Dataset:
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