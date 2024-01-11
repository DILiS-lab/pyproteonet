from typing import Optional

import numpy as np

from ..data.dataset import Dataset
from ..processing.molecule_set_transforms import num_theoretical_peptides
from .partner_summarization import partner_aggregation


def iBAQ(
    dataset: Dataset,
    mapping: str,
    partner_column: str,
    sequence_column: str,
    protein_molecule: str = "protein",
    result_column: Optional[str] = None,
    enzyme: str = "Trypsin",
    min_peptide_length: int = 6,
    max_peptide_length: int = 30,
    only_unique: bool = True,
    is_log: bool = False
):
    """
    Runs iBAQ aggregation on the dataset.

    Args:
        dataset (Dataset): Dataset to run aggregation on.
        mapping (str): Mapping to use (should represent peptide-to-protein mapping).
        partner_column (str): Name of the column with peptide abundance values.
        sequence_column (str): Name of the column in the MoleculeSet of the dataset containing protein sequences.
        result_column (str, optional): Name of the column to store the result in. Defaults to None.
        enzyme (str, optional): Enzyme to use for splitting protein sequence into peptides. Defaults to "Trypsin".
        min_peptide_length (int, optional): Minimum length of peptides to consider. Defaults to 6.
        max_peptide_length (int, optional): Maximum length of peptides to consider. Defaults to 30.
        only_unique (bool, optional): Whether to consider only unique peptides. Defaults to True.
        is_log (bool, optional): Whether the values in the dataset are log-transformed. Defaults to False.
    """
    num_peptides = num_theoretical_peptides(
        molecule_set=dataset.molecule_set,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        enzyme=enzyme,
        sequence_column=sequence_column,
        result_column=None,
    )
    res = partner_aggregation(
        dataset=dataset,
        molecule=protein_molecule,
        only_unique=only_unique,
        partner_column=partner_column,
        mapping=mapping,
        result_column=result_column,
        is_log=is_log
    )
    if is_log:
        res = np.exp(res)
    res /= num_peptides.loc[res.index.get_level_values('id')].values
    mask = res == np.inf
    res[mask] = dataset.missing_value
    if is_log:
        res = np.log(res)
    if result_column is not None:
        dataset.values[protein_molecule][result_column] = res
    return res
