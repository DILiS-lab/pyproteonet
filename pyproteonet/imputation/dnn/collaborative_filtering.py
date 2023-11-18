from typing import Optional
from tempfile import TemporaryDirectory

import pandas as pd

from ..vaep.sklearn.cf_transformer import CollaborativeFilteringTransformer

def impute_collaborative_filtering(dataset, molecule: str, column: str, result_column: Optional[str] = None)->pd.DataFrame:
    in_df = dataset.values[molecule][column]
    in_df = pd.DataFrame({'abundance':in_df})
    in_df_non_na = in_df[~in_df.abundance.isna()]
    with TemporaryDirectory() as out_dir:
        cf_model = CollaborativeFilteringTransformer(
            target_column='abundance',
            sample_column='sample',
            item_column='id',
            out_folder=out_dir,
            batch_size=min(4096, len(in_df_non_na)))
        cf_model.fit(in_df_non_na,
                     cuda=False,
                     epochs_max=20,
                     )
        print(in_df_non_na.shape)
        df_imputed = cf_model.transform(in_df_non_na)
        print(df_imputed.shape)
        if result_column is not None:
            dataset.values[molecule][result_column] = df_imputed
        assert df_imputed.isna().sum() == 0
        return df_imputed