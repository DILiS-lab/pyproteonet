from typing import Union, Callable, Optional

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy

from ..utils.numpy import eq_nan
from ..data.dataset import Dataset


def _get_mapped(
    dataset: Dataset,
    molecule: str,
    mapping: str,
    partner_column: str,
    only_unique: bool = True,
) -> pd.DataFrame:
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    mapped = dataset.get_mapped(
        molecule=molecule, partner_columns=[partner_column], mapping=mapping
    )
    mapped.rename(columns={partner_column: "quanti"}, inplace=True)
    degs = dataset.molecule_set.get_mapping_degrees(
        molecule=partner_molecule, mapping=mapping
    )
    mapped["deg"] = degs.loc[
        mapped.index.get_level_values(level=partner_molecule)
    ].values
    mapped = mapped[~eq_nan(mapped.quanti, dataset.missing_value)]
    if only_unique:
        mapped = mapped[mapped.deg == 1]
    return mapped


def partner_aggregation(
    dataset: Dataset,
    molecule: str,
    partner_column: str,
    mapping: str,
    method: Union[Callable[[SeriesGroupBy], pd.Series], str] = "sum",
    only_unique: bool = True,
    result_column: Optional[str] = None,
    is_log: bool = False,
) -> pd.Series:
    """Aggregates the values of partner molecule (e.g. peptides) into a molecule value. Partner molecule are determined according to a mapping.

    Args:
        dataset (Dataset): Dataset to run aggregation on.
        molecule (str): The molecule values should be aggregated for (e.g. protein).
        partner_column (str): The columns of the partner molecule containing abundance values that should be aggregated.
        mapping (str): Either the name of the aggregated molecule (e.g. peptide) or the name of the mapping linking the molecule from above to a partner molecule (e.g. peptide-protein mapping)
        method (Union[Callable[[SeriesGroupBy], pd.Series], str], optional): Either a function that takes a pandas groupby object and returns a series or one of the following strings: 'mean', 'sum', 'median', 'min', 'max'. Defaults to 'sum'.
        only_unique (bool, optional): Only consider unique peptides and ignore shared peptides. Defaults to True.
        result_column (Optional[str], optional): If given aggregation results are stored in this alue column of the molecule. Defaults to None.
        is_log (bool, optional): Wheter the input values are logarithmized. Defaults to False.

    Returns:
        pd.Series: A pandas series with sample id and molecule id as multiindex containing the aggregated values.
    """
    if isinstance(method, str):
        method = method.lower()
        if method == "mean":
            ag_fn = lambda group: group.mean()
        if method == "sum":
            ag_fn = lambda group: group.sum()
        elif method == "median":
            ag_fn = lambda group: group.median()
        elif method == "min":
            ag_fn = lambda group: group.min()
        elif method == "max":
            ag_fn = lambda group: group.max()
        else:
            raise AttributeError(f"Aggregation function {method} not known.")
    else:
        ag_fn = method
    mapped = _get_mapped(
        dataset=dataset,
        molecule=molecule,
        mapping=mapping,
        partner_column=partner_column,
        only_unique=only_unique,
    )
    if is_log:
        mapped["quanti"] = np.exp(mapped["quanti"])
    group = mapped.quanti.groupby(["sample", molecule])
    res = ag_fn(group)
    res.index.set_names("id", level=1, inplace=True)
    if is_log:
        res = np.log(res)
    if result_column is not None:
        dataset.values[molecule][result_column] = res
    return res


def partner_top_n_mean(
    dataset: Dataset,
    molecule: str,
    mapping: str,
    partner_column: str,
    top_n: int = 3,
    only_unique: bool = True,
    result_column: Optional[str] = None,
    skip_if_less_than_n: bool = True,
    is_log: bool = False,
) -> Optional[pd.Series]:
    """Aggregates the values of partner molecule (e.g. peptides) into a molecule (e.g. protein) value by averaging the abundance values of the three highest abundant partner molecules. Partner molecule are determined according to a mapping.

    Args:
        dataset (Dataset): Dataset to run aggregation on.
        molecule (str): The molecule values should be aggregated for (e.g. protein).
        partner_column (str): The columns of the partner molecule containing abundance values that should be aggregated.
        mapping (str): Either the name of the aggregated molecule (e.g. peptide) or the name of the mapping linking the molecule from above to a partner molecule (e.g. peptide-protein mapping)
        top_n (int, optional): Number of highest abundant partner molecules to consider. Defaults to 3.
        only_unique (bool, optional): Only consider unique peptides and ignore shared peptides. Defaults to True.
        result_column (Optional[str], optional): If given aggregation results are stored in this alue column of the molecule. Defaults to None.
        skip_if_less_than_n (bool, optional): If True, the aggregation is skipped if there are less than n partner molecules and the aggregated values is set to NaN. Defaults to True.
        is_log (bool, optional): Wheter the input values are logarithmized. Defaults to False.

    Returns:
        pd.Series: A pandas series with sample id and molecule id as multiindex containing the aggregated values.
    """
    mapped = _get_mapped(
        dataset=dataset,
        molecule=molecule,
        mapping=mapping,
        partner_column=partner_column,
        only_unique=only_unique,
    )
    if is_log:
        mapped["quanti"] = np.exp(mapped["quanti"])
    group = (
        mapped.quanti.sort_values(ascending=False)
        .groupby(["sample", molecule])
        .head(top_n)
        .groupby(["sample", molecule])
    )
    if skip_if_less_than_n:
        res = group.mean()[group.count() >= top_n]
    else:
        res = group.mean()
    res.index.set_names("id", level=1, inplace=True)
    if is_log:
        res = np.log(res)
    if result_column is not None:
        dataset.set_column_lf(
            molecule=molecule, values=res, column=result_column, fill_missing=True
        )
    return res
