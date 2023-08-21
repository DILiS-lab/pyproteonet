from typing import Optional, List, Tuple
import functools
import warnings

import pandas as pd
import alphapept.quantification
from alphapept.interface import tqdm_wrapper
from tqdm.auto import tqdm

from ..data.dataset import Dataset


def _dataset_to_alphapept_frame(
    dataset: Dataset,
    molecule: str = "peptide",
    quantity_column: str = "abundance",
    mapping_molecule: Optional[str] = None,
    mapping: Optional[str] = None,
    skip_na: bool = True,
):
    df = pd.DataFrame({"quantity": dataset.get_column_flat(molecule=molecule, column=quantity_column)})
    # make conform with alphapept format so we can use alphapept's MaxLFQ implementation
    df.reset_index(inplace=True)
    df["fraction"] = 1  # TODO: think about supporting fractions
    if skip_na:
        df = df[~df.quantity.isna()]
    if mapping_molecule is not None and mapping is not None:
        map_pairs = dataset.molecule_set.get_mapped_pairs(
            molecule_a=molecule, molecule_b=mapping_molecule, mapping=mapping
        )
        # map_pairs.set_index(molecule, verify_integrity=True, inplace=True)
        df = df.merge(map_pairs[[molecule, mapping_molecule]], left_on="id", right_on=molecule, how="inner")
        del df[molecule]
        # df["protein_group"] = map_pairs.loc[df.id][mapping_molecule].values
    df.rename(columns={"sample": "sample_group", "id": "precursor", mapping_molecule: "protein_group"}, inplace=True)
    return df


def delayed_normalization(
    dataset: Dataset,
    molecule: str = "peptide",
    column: str = "abundance",
    result_column: Optional[str] = None,
    inplace: bool = False,
) -> Tuple[Dataset, List[float]]:
    """Applies the normalization from the MaxLFQ paper (using the alphapept implemetation)

    Args:
        dataset (Dataset): Dataset to normalize.
        molecule (str, optional): The molecule in the dataset whose values will be normalized. Defaults to "peptide".
        column (str, optional): The value column to normalize. Defaults to "abundance".
        result_column (Optional[str], optional): The column to write the results to. If not given defaults to column.
        inplace (bool, optional): Whether to copy the dataset. Defaults to False.

    Returns:
        Tuple[Dataset, List[float]]: Tuple of the resulting dataset and the normalization factors.
    """
    if result_column is None:
        result_column = column
    if not inplace:
        dataset = dataset.copy()
    df = _dataset_to_alphapept_frame(dataset=dataset, quantity_column=column, molecule=molecule, skip_na=True)
    df, normalization_factors = alphapept.quantification.delayed_normalization(df, "quantity")

    df_indexed = df.rename(columns={"precursor": "id", "sample_group": "sample"})
    df_indexed.set_index(["sample", "id"], inplace=True)
    dataset.set_column_flat(molecule=molecule, values=df_indexed["quantity_dn"], column=result_column)
    return dataset, normalization_factors


def max_lfq(
    dataset: Dataset,
    molecule="peptide",
    column="abundance",
    result_molecule="protein",
    result_column=None,
    mapping="protein",
    normalize: bool = True,
    only_unique=True,
    minimum_ratios: int = 1,
    inplace: bool = False,
    pbar: bool = False,
) -> Dataset:
    if result_column is None:
        result_column = column
    if not inplace:
        dataset = dataset.copy()
    samples = dataset.sample_names
    df = _dataset_to_alphapept_frame(
        dataset=dataset,
        molecule=molecule,
        quantity_column=column,
        mapping_molecule=result_molecule,
        mapping=mapping,
        skip_na=True,
    )
    if only_unique:
        uniques = dataset.molecule_set.get_mapping_unique_molecules(
            molecule=molecule, partner_molecule=result_molecule, mapping=mapping
        )
        df = df[df["precursor"].isin(uniques)]
    # df_grouped = df.groupby(['sample_group', 'precursor', 'protein_group'])[['quantity_dn']].sum().reset_index()
    if normalize:
        df, normalization_factors = alphapept.quantification.delayed_normalization(df, "quantity")
        del df["quantity"]
        df.rename(columns={"quantity_dn": "quantity"}, inplace=True)
    cb = None
    if pbar:
        cb = functools.partial(tqdm_wrapper, tqdm(total=1))
    with warnings.catch_warnings():
        # alphapept uses the np.nanmean function which throw a warning if all values are nan and returns nan
        # since this is expected behavior we ignore those warnings
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        protein_table = alphapept.quantification.protein_profile_parallel(
            field="quantity", minimum_ratios=minimum_ratios, df=df, callback=cb
        )
    lfq_samples = [s + "_LFQ" for s in samples]
    df = protein_table.loc[:, lfq_samples]
    df.rename(columns={lfq_s: s for lfq_s, s in zip(lfq_samples, samples)}, inplace=True)
    df = df.stack()
    df.index = df.index.rename(names=["id", "sample"])
    dataset.set_column_flat(molecule=result_molecule, values=df, column=result_column)
    return dataset
