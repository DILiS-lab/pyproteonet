from typing import Optional, Literal
from tempfile import TemporaryDirectory

import pandas as pd
import torch

from ...data.dataset import Dataset
from ..vaep.sklearn.ae_transformer import AETransformer
from ..vaep.sampling import sample_data


def impute_auto_encoder(
    dataset: Dataset,
    molecule: str,
    column: str,
    result_column: Optional[str] = None,
    validation_fraction: float = 0.1,
    batch_size: int = 26,
    model_type: Literal["VAE", "DAE"] = "VAE",
    cuda: Optional[bool] = None
) -> pd.DataFrame:
    if cuda is None:
        cuda = torch.cuda.is_available()
    df = dataset.get_samples_value_matrix(molecule=molecule, column=column)
    df.index.name = "id"
    df.columns.name = "sample"
    in_df = df
    freq_feat = in_df.notna().sum()
    val_X, train_X = sample_data(
        in_df.stack(),
        sample_index_to_drop=0,
        weights=freq_feat,
        frac=validation_fraction,
        random_state=42,
    )
    val_X, train_X = val_X.unstack(), train_X.unstack()
    val_X = pd.DataFrame(pd.NA, index=train_X.index, columns=train_X.columns).fillna(
        val_X
    )
    while train_X.shape[0] % batch_size == 1:
        batch_size += 1
    with TemporaryDirectory() as out_dir:
        model = AETransformer(
            model=model_type,
            hidden_layers=[
                512,
            ],
            latent_dim=50,
            out_folder=out_dir,
            batch_size=batch_size,
            cuda = cuda
        )
        model.fit(train_X, val_X, epochs_max=50, cuda=False)
        df_imputed = model.transform(in_df)
        mask = ~df.isna()
        df_imputed[mask] = df[mask]
        df_imputed = df_imputed.stack().swaplevel()
        if result_column is not None:
            dataset.values[molecule][result_column] = df_imputed
        assert df_imputed.isna().sum() == 0
        return df_imputed
