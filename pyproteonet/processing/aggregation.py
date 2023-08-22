from typing import Callable, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy
from tqdm.auto import tqdm

from ..data.dataset import Dataset
from ..utils.numpy import eq_nan


def aggregate_peptides(
    dataset: Dataset,
    aggregation_fn: Callable[[SeriesGroupBy], pd.Series],
    input_molecule: str = 'peptide',
    column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: Optional[str] = None,
    mapping: str = "protein",
    only_unique: bool = True,
    inplace: bool = False,
    tqdm_bar: bool = False,
) -> Optional[Dataset]:
    if not inplace:
        dataset = dataset.copy()
    if result_column is None:
        result_column = column
    mapped = dataset.molecule_set.get_mapped_pairs(molecule_a=result_molecule, molecule_b=input_molecule, mapping=mapping)
    unique_peptides = []
    if only_unique:
        unique_peptides = mapped.groupby(input_molecule)[result_molecule].count()
        unique_peptides = unique_peptides[unique_peptides == 1].index
    for sample in tqdm(dataset.samples) if tqdm_bar else dataset.samples:
        sample_mapping = mapped.copy()
        sample_mapping["val"] = (
            sample.values[input_molecule].loc[sample_mapping[input_molecule], column].to_numpy()
        )
        sample_mapping["missing"] = (
            sample.missing_mask(molecule=input_molecule, column=column).loc[sample_mapping[input_molecule]].to_numpy()
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


def neighbor_mean(
    dataset: Dataset,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = 'abundance',
    result_molecule: str = 'protein',
    result_column: str = None,
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
        input_molecule=molecule,
        column=column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def neighbor_median(
    dataset: Dataset,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = 'abundance',
    result_molecule: str = 'protein',
    result_column: str = None,
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
        input_molecule=molecule,
        column=column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def neighbor_sum(
    dataset: Dataset,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: Optional[str] = None,
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
        input_molecule=molecule,
        column=column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def neighbor_min(
    dataset: Dataset,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = None,
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
        input_molecule=molecule,
        column=column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def neighbor_max(
    dataset: Dataset,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = None,
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
        input_molecule=molecule,
        column=column,
        result_molecule=result_molecule,
        result_column=result_column,
        mapping=mapping,
        inplace=inplace,
        tqdm_bar=tqdm_bar,
    )


def neighbor_top_n_mean(
    dataset: Dataset,
    top_n: int = 3,
    only_unique: bool = True,
    molecule: str = 'peptide',
    column: str = "abundance",
    result_molecule: str = 'protein',
    result_column: str = None,
    mapping: str = None,
    inplace: bool = False,
):
    if not inplace:
        dataset = dataset.copy()
    mapped = dataset.get_mapped(molecule=molecule, partner_molecule=result_molecule, columns=[column], mapping=mapping)
    mapped.rename(columns={column:'quanti'}, inplace=True)
    if mapping is None:
        mapping = result_molecule
    degs = dataset.molecule_set.get_mapping_degrees(molecule=molecule, mapping=mapping, partner_molecule=result_molecule)
    mapped['deg'] = degs.loc[mapped.index.get_level_values(level=molecule)].values
    mapped = mapped[~eq_nan(mapped.quanti, dataset.missing_value)]
    if only_unique:
        mapped = mapped[mapped.deg==1]
    group = mapped.quanti.sort_values(ascending=False).groupby(['sample', result_molecule]).head(top_n).groupby(['sample',result_molecule])
    res = group.mean()[group.count() >= top_n]
    dataset.values[result_molecule][result_column] = res
    if not inplace:
        return dataset
    
    
