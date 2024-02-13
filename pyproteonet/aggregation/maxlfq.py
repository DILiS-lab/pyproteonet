'''
Mostly taken from https://github.com/InfectionMedicineProteomics/DPKS/
'''

from typing import Optional
import math

import numba  # type: ignore
import numpy as np
import pandas as pd
from numba import njit, prange  # type: ignore
from numba_progress import ProgressBar

from ..data.dataset import Dataset


@njit(nogil=True)
def get_ratios(quantitative_data, sample_combinations, min_ratios: int):
    num_samples = quantitative_data.shape[1]

    ratios = np.empty((num_samples, num_samples), dtype=np.float64)

    ratios[:] = np.nan

    for combination in sample_combinations:
        sample_a = combination[0]
        sample_b = combination[1]

        ratio = -quantitative_data[:, sample_a] + quantitative_data[:, sample_b]

        ratio_median = np.nanmedian(ratio)
        if min_ratios > 1 and (~np.isnan(ratio)).sum() < min_ratios:
            ratio_median = np.nan

        ratios[sample_b, sample_a] = ratio_median
    return ratios


@njit(nogil=True)
def solve_profile(X, ratios, sample_combinations):
    if np.all(np.isnan(X)):
        results = np.zeros((X.shape[1]))

    else:
        num_samples = X.shape[1]

        A = np.zeros((num_samples + 1, num_samples + 1))
        b = np.zeros((num_samples + 1,))

        for sample_combination in sample_combinations:
            i = sample_combination[0]
            j = sample_combination[1]


            ratio = ratios[j, i]
            if not np.isnan(ratio):
                A[i][j] = -1.0
                A[j][i] = -1.0
                A[i][i] += 1.0
                A[j][j] += 1.0
                b[i] -= ratio
                b[j] += ratio

        formatted_a = 2.0 * A
        formatted_a[:num_samples, num_samples] = 1
        formatted_a[num_samples, :num_samples] = 1

        formatted_b = 2.0 * b

        sample_mean = np.nanmean(X)

        if np.isnan(sample_mean):
            sample_mean = 0.0

        formatted_b[num_samples] = sample_mean * num_samples

        nan_idx = np.argwhere(np.isnan(b))

        for nan_value in nan_idx:
            formatted_b[nan_value] = 0.0

        results = np.linalg.lstsq(formatted_a, formatted_b, -1.0)[0][: X.shape[1]]

    results[results == 0.0] = np.nan

    return results


@njit(nogil=True)
def build_connection_graph(grouping, min_ratios: int):
    connected_sample_groups = numba.typed.List()
    num_samples = grouping.shape[1]

    adj = numba.typed.List([numba.typed.List.empty_list(numba.int64) for x in range(num_samples)])
    for sample_idx in range(num_samples):
        for compared_sample_idx in range(num_samples):
            if sample_idx == compared_sample_idx:
                continue
            ratio = -grouping[:, sample_idx] + grouping[:, compared_sample_idx]
            if (~np.isnan(ratio)).sum() >= min_ratios:
                adj[sample_idx].append(compared_sample_idx)
    visited = np.full(num_samples, False)
    i = 0
    for sample_id in range(num_samples):
        if visited[sample_id]:
            continue
        sample_group = numba.typed.List.empty_list(numba.int64)
        q = numba.typed.List.empty_list(numba.int64)
        q.append(sample_id)
        visited[sample_id] = True
        while len(q):
            i+=1
            current = q.pop()
            sample_group.append(current)
            for neighbor in adj[current]:
                if not visited[neighbor]:
                    q.append(neighbor)
                    visited[neighbor] = True
        connected_sample_groups.append(np.asarray(sample_group))
    return connected_sample_groups


@njit(nogil=True)
def build_combinations(subset):
    column_idx = np.arange(0, subset.shape[1])

    combos = []

    for i in column_idx:
        for j in range(i + 1, column_idx.shape[0]):
            combos.append([i, j])

    return np.array(combos)


@njit(nogil=True)
def mask_group(grouping):
    nan_groups = []

    for subgroup_idx in range(grouping.shape[0]):
        if not np.isnan(grouping[subgroup_idx, :]).all():
            nan_groups.append(subgroup_idx)

    grouping = grouping[np.array(nan_groups, dtype=np.uint64), :]

    return grouping


@njit(nogil=True)
def quantify_group(grouping, connected_graph, min_ratios: int, median_fallback: bool):
    profile = np.zeros((grouping.shape[1]))

    for graph in connected_graph:
        if graph.shape[0] == 1:
            subset = grouping[:, graph]

            if np.isnan(subset).all() or not median_fallback:
                profile[graph] = np.nan

            else:
                profile[graph] = np.nanmedian(subset)

        if graph.shape[0] > 1:
            subset = grouping[:, graph]

            sample_combinations = build_combinations(subset)

            ratios = get_ratios(subset, sample_combinations, min_ratios=min_ratios)

            solved_profile = solve_profile(subset, ratios, sample_combinations)

            for results_idx in range(solved_profile.shape[0]):
                profile[graph[results_idx]] = solved_profile[results_idx]

    return profile


@njit(parallel=True)
def quantify_groups(groupings, minimum_subgroups, min_ratios: int, median_fallback: bool, pbar: Optional[ProgressBar] = None):
    num_groups = len(groupings)

    results = np.empty(shape=(num_groups, groupings[0].shape[1]))

    for group_idx in prange(num_groups):
        grouping = mask_group(groupings[group_idx])

        if grouping.shape[0] >= minimum_subgroups:
            connected_graph = build_connection_graph(grouping=grouping, min_ratios=min_ratios)

            profile = quantify_group(grouping, connected_graph, min_ratios=min_ratios, median_fallback=median_fallback)

        else:
            profile = np.zeros((grouping.shape[1]))
            profile[:] = np.nan

        for sample_idx in range(profile.shape[0]):
            results[group_idx, sample_idx] = profile[sample_idx]
        if pbar is not None:
            pbar.update(1)
    return results


def maxlfq(dataset: Dataset, molecule: str, mapping: str, partner_column: str, min_subgroups: int = 1, min_ratios: int = 1, median_fallback: bool = True,
           is_log: bool = False, only_unique: bool = True, result_column: Optional[str] = None, pbar: bool = False)->pd.Series:
    """Runs MaxLFQ aggregation on the dataset.

    Args:
        dataset (Dataset): Dataset to run aggregation on.
        molecule (str): The molecule values should be aggregated for (e.g. protein).
        mapping (str): Either the name of the aggregated molecule (e.g. peptide) or the name of the mapping linking the molecule from above to a partner molecule (e.g. peptide-protein mapping)
        partner_column (str): The columns of the partner molecule containing abundance values that should be aggregated.
        min_subgroups (int, optional): Minimum number of partner molecules required otherwise the aggregation result is set to NaN. Defaults to 1.
        min_ratios (int, optional): Minimum number of ratios rquired to generate an aggregated value, otherwise the aggregating result is set to NaN. Defaults to 1.
        median_fallback (bool, optional): Fallback to median abundance values if no ratios can be inferred. Defaults to True.
        is_log (bool, optional): Wheter the input values are logarithmized. Defaults to False.
        only_unique (bool, optional): Only consider unique peptides and ignore shared peptides. Defaults to True.
        result_column (Optional[str], optional): If given aggregation results are stored in this alue column of the molecule. Defaults to None.
        pbar (bool, optional): Wheter to display a progress bar. Defaults to False.

    Returns:
        pd.Series: A pandas series with sample id and molecule id as multiindex containing the aggregated values.
    """
    mapped = dataset.molecule_set.get_mapped(molecule=molecule, mapping=mapping)
    molecule, mapping, partner = dataset.infer_mapping(molecule=molecule, mapping=mapping)
    degs = dataset.molecule_set.get_mapping_degrees(molecule=partner, mapping=mapping)
    if only_unique:
        considered_partners = degs[degs==1].index
    else:
        considered_partners = degs.index
    mapped = mapped[mapped.index.get_level_values(partner).isin(considered_partners)]
    mat = dataset.get_samples_value_matrix(molecule=partner, column=partner_column)
    if not is_log:
        mat = np.log(mat)
    groupings_dict = {mol:np.ascontiguousarray(mat.loc[partner_ids.index.get_level_values(partner),:].to_numpy().astype(np.float64))
                      for mol,partner_ids in mapped.groupby(molecule)}
    groupings = numba.typed.List()
    group_ids = []
    for key, group in groupings_dict.items():
        groupings.append(group)
        group_ids.append(key)
    res_mat = pd.DataFrame(np.nan, index=dataset.molecules[molecule].index, columns=mat.columns)
    progress_bar = None
    if pbar:
        progress_bar = ProgressBar(total=len(groupings))
    res = quantify_groups(groupings=groupings, minimum_subgroups=min_subgroups, min_ratios=min_ratios, median_fallback=median_fallback,
                          pbar=progress_bar)
    if pbar:
        progress_bar.close()
    if not is_log:
        res = math.e ** res
    res_mat.loc[group_ids, :] = res
    if result_column is not None:
        dataset.set_wf(matrix=res_mat, molecule=molecule, column=result_column)
    vals = res_mat.stack(dropna=False).swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals