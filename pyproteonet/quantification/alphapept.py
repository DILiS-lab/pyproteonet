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
    molecule: str,
    mapping: str,
    partner_column: str = "abundance",
    skip_na: bool = True,
):
    partner_molecule = dataset.get_mapping_partner(molecule=molecule, mapping=mapping)
    df = pd.DataFrame({"quantity": dataset.get_column_flat(molecule=partner_molecule, column=partner_column)})
    # make conform with alphapept format so we can use alphapept's MaxLFQ implementation
    df.reset_index(inplace=True)
    df["fraction"] = 1  # TODO: think about supporting fractions
    if skip_na:
        df = df[~df.quantity.isna()]
    map_pairs = dataset.molecule_set.get_mapped_pairs(
        molecule_a=partner_molecule, molecule_b=molecule, mapping=mapping
    )
    # map_pairs.set_index(molecule, verify_integrity=True, inplace=True)
    df = df.merge(map_pairs[[partner_molecule, molecule]], left_on="id", right_on=partner_molecule, how="inner")
    del df[partner_molecule]
    # df["protein_group"] = map_pairs.loc[df.id][mapping_molecule].values
    df.rename(columns={"sample": "sample_group", "id": "precursor", molecule: "protein_group"}, inplace=True)
    return df

def max_lfq(
    dataset: Dataset,
    molecule: str,
    mapping: str,
    partner_column: str,
    normalize: bool = False,
    only_unique=True,
    minimum_ratios: int = 1,
    result_column: Optional[str]=None,
    pbar: bool = False,
) -> pd.Series:
    if result_column is None:
        result_column = partner_column
    samples = dataset.sample_names
    df = _dataset_to_alphapept_frame(
        dataset=dataset,
        molecule=molecule,
        mapping=mapping,
        partner_column=partner_column,
        skip_na=True,
    )
    partner_molecule = dataset.get_mapping_partner(molecule=molecule, mapping=mapping)
    if only_unique:
        uniques = dataset.molecule_set.get_mapping_unique_molecules(
            molecule=partner_molecule, partner_molecule=molecule, mapping=mapping
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
    df = df.stack().swaplevel()
    df.index.rename(names=["sample", "id"], inplace=True)
    if result_column is not None:
        dataset.set_column_flat(molecule=molecule, values=df, column=result_column)
    return df

