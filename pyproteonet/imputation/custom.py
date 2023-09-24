from ..data.dataset import Dataset
from ..quantification.neighbor_summarization import neighbor_top_n_mean, neighbor_aggregation

import pandas as pd

def multistep_impute(dataset: Dataset, molecule: str, column: str, mapping: str, partner_column: str, all_missing_percentile: float = 0.1)->pd.Series:
    top3 = neighbor_top_n_mean(dataset=dataset, molecule=molecule, partner_column=partner_column, mapping=mapping, top_n=3, only_unique=True, 
                               skip_if_less_than_n=False)
    median = neighbor_aggregation(dataset=dataset, molecule=molecule, partner_column=partner_column, mapping=mapping, only_unique=False, method='median')
    myimp = dataset.values[molecule][column]
    missing_mask = myimp.isna()
    myimp[missing_mask] = top3[top3.index.isin(missing_mask[missing_mask].index)]
    #missing_mask = myimp.isna()
    #myimp[missing_mask] = top1[top1.index.isin(missing_mask[missing_mask].index)]
    missing_mask = myimp.isna()
    myimp[missing_mask] = median[median.index.isin(missing_mask[missing_mask].index)]
    myimp[myimp.isna()] = dataset.values[molecule][column].quantile(all_missing_percentile / 100)
    assert myimp.isna().sum()==0
    return myimp