from ..data.dataset import Dataset
from ..aggregation.partner_summarization import partner_top_n_mean, partner_aggregation

import pandas as pd


def multistep_top3_impute(
    dataset: Dataset,
    molecule: str,
    column: str,
    mapping: str,
    partner_column: str,
    all_missing_percentile: float = 0.1,
) -> pd.Series:
    """Simple imputation method based on top3 aggregation using other aggregation approaches as fallback options.

    Args:
        dataset (Dataset): The dataset to be imputed.
        molecule (str): The molecule type to be imputed.
        column (str): The value column with missing values to be imputed. 
        mapping (str): The mapping to use for aggregation.
        partner_column (str): The partner column to use for aggregation.
        all_missing_percentile (float, optional): The percentile to use as fallback value when aggregation fails. Defaults to 0.1.

    Returns:
        pd.Series: _description_
    """
    top3 = partner_top_n_mean(
        dataset=dataset,
        molecule=molecule,
        partner_column=partner_column,
        mapping=mapping,
        top_n=3,
        only_unique=True,
        skip_if_less_than_n=False,
    )
    median = partner_aggregation(
        dataset=dataset,
        molecule=molecule,
        partner_column=partner_column,
        mapping=mapping,
        only_unique=False,
        method="median",
    )
    myimp = dataset.values[molecule][column]
    missing_mask = myimp.isna()
    myimp[missing_mask] = top3[top3.index.isin(missing_mask[missing_mask].index)]
    # missing_mask = myimp.isna()
    # myimp[missing_mask] = top1[top1.index.isin(missing_mask[missing_mask].index)]
    missing_mask = myimp.isna()
    myimp[missing_mask] = median[median.index.isin(missing_mask[missing_mask].index)]
    myimp[myimp.isna()] = dataset.values[molecule][column].quantile(
        all_missing_percentile / 100
    )
    assert myimp.isna().sum() == 0
    return myimp
