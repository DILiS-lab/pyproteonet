from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from tqdm.auto import tqdm

from ..data.dataset import Dataset
from .molecule_set_transforms import num_theoretical_peptides


def aggregate_peptides(
    dataset: Dataset,
    aggregation_fn: Callable[[SeriesGroupBy], pd.Series],
    only_unique: bool = True,
    input_column: str = "abundance",
    result_column: str = "abundance",
    mapping: str = "protein",
    inplace: bool = False,
    tqdm_bar: bool = False,
) -> Optional[Dataset]:
    if not inplace:
        dataset = dataset.copy()
    mapped = dataset.molecule_set.get_mapped_pairs("protein", "peptide", mapping=mapping)
    unique_peptides = []
    if only_unique:
        unique_peptides = mapped.groupby("peptide").protein.count()
        unique_peptides = unique_peptides[unique_peptides == 1].index
    for sample in tqdm(dataset.samples) if tqdm_bar else dataset.samples:
        sample_mapping = mapped.copy()
        sample_mapping["peptide_abundance"] = (
            sample.values["peptide"].loc[sample_mapping.peptide, input_column].to_numpy()
        )
        sample_mapping["missing"] = (
            sample.missing_mask(molecule="peptide", column=input_column).loc[sample_mapping.peptide].to_numpy()
        )
        sample_mapping = sample_mapping[~sample_mapping["missing"]]
        # sample_mapping.sort_values('peptide_abundance', inplace=True, ascending=False)
        if only_unique:
            sample_mapping = sample_mapping[sample_mapping.peptide.isin(unique_peptides)]
        groups = sample_mapping.groupby("protein").peptide_abundance
        res = aggregation_fn(groups)
        sample.values["protein"].loc[:, result_column] = dataset.missing_value
        sample.values["protein"].loc[res.index, result_column] = res
    if not inplace:
        return dataset


def mean(
    dataset: Dataset,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def median(
    dataset: Dataset,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def sum(
    dataset: Dataset,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def minimum(
    dataset: Dataset,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def maximum(
    dataset: Dataset,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def top_n_mean(
    dataset: Dataset,
    top_n: int = 3,
    only_unique: bool = True,
    input_column: str = "abundance",
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
        input_column=input_column,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def iBAQ(
    dataset: Dataset,
    input_column: str = 'abundance',
    sequence_column: str = "sequence",
    enzyme: str = "Trypsin",
    min_peptide_length: int = 6,
    max_peptide_length: int = 30,
    result_column: str = "abundance",
    only_unique: bool = False,
    mapping: str = 'protein',
    inplace: bool = False,
    tqdm_bar: bool = False
):
    num_peptides = num_theoretical_peptides(molecule_set=dataset.molecule_set, min_peptide_length=min_peptide_length, max_peptide_length=max_peptide_length,
        enzyme=enzyme, sequence_column=sequence_column, result_column=None)
    res = sum(dataset=dataset, only_unique=only_unique, input_column=input_column, result_column=result_column, mapping=mapping, inplace=inplace, tqdm_bar=tqdm_bar)
    if not inplace:
        dataset = res # type: ignore
    for sample in dataset.samples:
        prot_vals = sample.values['protein']
        prot_vals[result_column] /= num_peptides.loc[prot_vals.index]
        mask = prot_vals[result_column] == np.inf
        prot_vals.loc[mask, result_column] = dataset.missing_value
    if not inplace:
        return dataset
    
    
