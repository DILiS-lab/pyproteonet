'''
Mostly taken from https://github.com/InfectionMedicineProteomics/DPKS/
'''

from typing import Optional
import math

import numba  # type: ignore
import numpy as np
# import numpy.typing as npt  # not used yet
import pandas as pd  # type: ignore
from numba import njit, prange  # type: ignore

from ..data.dataset import Dataset


@njit(nogil=True)
def get_ratios(quantitative_data, sample_combinations):
    num_samples = quantitative_data.shape[1]

    ratios = np.empty((num_samples, num_samples), dtype=np.float64)

    ratios[:] = np.nan

    for combination in sample_combinations:
        sample_a = combination[0]
        sample_b = combination[1]

        ratio = -quantitative_data[:, sample_a] + quantitative_data[:, sample_b]

        ratio_median = np.nanmedian(ratio)

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

            A[i][j] = -1.0
            A[j][i] = -1.0
            A[i][i] += 1.0
            A[j][j] += 1.0

            ratio = ratios[j, i]

            if not np.isnan(ratio):
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
def build_connection_graph(grouping):
    connected_sample_groups = numba.typed.Dict()

    connected_indices = numba.typed.List()

    sample_group_id = 0

    for sample_idx in range(grouping.shape[1]):
        if sample_idx not in connected_indices:
            sample_group = []

            for compared_sample_idx in range(grouping.shape[1]):
                comparison = grouping[:, sample_idx] - grouping[:, compared_sample_idx]

                if not np.isnan(comparison).all():
                    sample_group.append(compared_sample_idx)

                    connected_indices.append(compared_sample_idx)

            if len(sample_group) > 0:
                connected_sample_groups[sample_group_id] = np.array(sample_group)

                sample_group_id += 1

            else:
                connected_sample_groups[sample_group_id] = np.array([sample_idx])

                sample_group_id += 1

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

    grouping = grouping[np.array(nan_groups), :]

    return grouping


@njit(nogil=True)
def quantify_group(grouping, connected_graph):
    profile = np.zeros((grouping.shape[1]))

    for sample_group_id, graph in connected_graph.items():
        if graph.shape[0] == 1:
            subset = grouping[:, graph]

            if np.isnan(subset).all():
                profile[graph] = np.nan

            else:
                profile[graph] = np.nanmedian(subset)

        if graph.shape[0] > 1:
            subset = grouping[:, graph]

            sample_combinations = build_combinations(subset)

            ratios = get_ratios(subset, sample_combinations)

            solved_profile = solve_profile(subset, ratios, sample_combinations)

            for results_idx in range(solved_profile.shape[0]):
                profile[graph[results_idx]] = solved_profile[results_idx]

    return profile


@njit(parallel=True)
def quantify_groups(groupings, minimum_subgroups):
    num_groups = len(groupings)

    results = np.empty(shape=(num_groups, groupings[0].shape[1]))

    for group_idx in prange(num_groups):
        grouping = mask_group(groupings[group_idx])

        if grouping.shape[0] >= minimum_subgroups:
            connected_graph = build_connection_graph(grouping)

            profile = quantify_group(grouping, connected_graph)

        else:
            profile = np.zeros((grouping.shape[1]))
            profile[:] = np.nan

        for sample_idx in range(profile.shape[0]):
            results[group_idx, sample_idx] = profile[sample_idx]

    return results


def maxlfq(dataset: Dataset, molecule: str, mapping: str, partner_column: str, is_log: bool = False, result_column: Optional[str] = None):
    mapped = dataset.molecule_set.get_mapped(molecule=molecule, mapping=mapping)
    molecule, mapping, partner = dataset.infer_mapping(molecule=molecule, mapping=mapping)
    degs = dataset.molecule_set.get_mapping_degrees(molecule=partner, mapping=mapping)
    unique_partners = degs[degs==1].index
    mapped = mapped[mapped.index.get_level_values(partner).isin(unique_partners)]
    mat = dataset.get_samples_value_matrix(molecule=partner, column=partner_column)
    if not is_log:
        mat = np.log(mat)
    groupings_dict = {mol:np.ascontiguousarray(mat.loc[partner_ids.index.get_level_values(partner),:].to_numpy().astype(float))
                    for mol,partner_ids in mapped.groupby(molecule)}
    groupings = numba.typed.List()
    group_ids = []
    for key, group in groupings_dict.items():
        groupings.append(group)
        group_ids.append(key)
    res_mat = pd.DataFrame(np.nan, index=dataset.molecules[molecule].index, columns=mat.columns)
    res = quantify_groups(groupings=groupings, minimum_subgroups=1)
    if not is_log:
        res = math.e ** res
    res_mat.loc[group_ids, :] = res
    if result_column is not None:
        dataset.set_samples_value_matrix(matrix=res_mat, molecule=molecule, column=result_column)
    vals = res_mat.stack().swaplevel()
    vals.index.set_names(["sample", "id"], inplace=True)
    return vals