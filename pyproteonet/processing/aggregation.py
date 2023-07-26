from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from tqdm.auto import tqdm

from ..data.dataset import Dataset


def aggregate_peptides(
    dataset: Dataset,
    aggregation_fn: Callable[[SeriesGroupBy], pd.Series],
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    only_unique: bool = True,
    inplace: bool = False,
    tqdm_bar: bool = False,
) -> Optional[Dataset]:
    if not inplace:
        dataset = dataset.copy()
    mapped = dataset.molecule_set.get_mapped_pairs(result_molecule, input_molecule, mapping=mapping)
    unique_peptides = []
    if only_unique:
        unique_peptides = mapped.groupby(input_molecule)[result_molecule].count()
        unique_peptides = unique_peptides[unique_peptides == 1].index
    for sample in tqdm(dataset.samples) if tqdm_bar else dataset.samples:
        sample_mapping = mapped.copy()
        sample_mapping["val"] = (
            sample.values[input_molecule].loc[sample_mapping[input_molecule], input_column].to_numpy()
        )
        sample_mapping["missing"] = (
            sample.missing_mask(molecule=input_molecule, column=input_column).loc[sample_mapping[input_molecule]].to_numpy()
        )
        sample_mapping = sample_mapping[~sample_mapping["missing"]]
        # sample_mapping.sort_values('val', inplace=True, ascending=False)
        if only_unique:
            sample_mapping = sample_mapping[sample_mapping[input_molecule].isin(unique_peptides)]
        groups = sample_mapping.groupby(result_molecule).val
        res = aggregation_fn(groups)
        sample.values[result_molecule][result_column] = dataset.missing_value
        sample.values[result_molecule][result_column] = res
    if not inplace:
        return dataset


def mean(
    dataset: Dataset,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _mean(groups: SeriesGroupBy) -> pd.Series:
        return groups.mean()

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_mean,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def median(
    dataset: Dataset,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _median(groups: SeriesGroupBy) -> pd.Series:
        return groups.median()

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_median,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def sum(
    dataset: Dataset,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _sum(groups: SeriesGroupBy) -> pd.Series:
        return groups.sum()

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_sum,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def minimum(
    dataset: Dataset,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _min(groups: SeriesGroupBy) -> pd.Series:
        return groups.min()

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_min,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def maximum(
    dataset: Dataset,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _max(groups: SeriesGroupBy) -> pd.Series:
        return groups.max()

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_max,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def top_n_mean(
    dataset: Dataset,
    top_n: int = 3,
    only_unique: bool = True,
    input_molecule: str = 'peptide',
    input_column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
):
    def _top_n_mean(groups: SeriesGroupBy) -> pd.Series:
        return groups.nlargest(top_n).groupby("protein").mean()[groups.count() >= top_n]

    return aggregate_peptides(
        dataset=dataset,
        aggregation_fn=_top_n_mean,
        only_unique=only_unique,
        input_molecule=input_molecule,
        input_column=input_column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )
    
    
