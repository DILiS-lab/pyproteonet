from typing import Optional, Callable, List, Dict

import numpy as np
import pandas as pd
import lightning.pytorch as pl

from ..data.dataset import Dataset
from ..predictors.gnn import GnnPredictor
from ..masking.masking import train_test_non_missing_no_overlap_iterable, non_missing_iterable
from ..masking.missing_values import mask_missing
from ..dgl.gnn_architectures.gat import GAT
from ..lightning.console_logger import ConsoleLogger
from ..data.masked_dataset import MaskedDataset


def gnn_impute(
    dataset: Dataset,
    molecule: str,
    column: str,
    mapping: str,
    result_column: str = "gnnimp",
    partner_molecule: Optional[str] = None,  # Deprecated, use mapping instead
    partner_column: Optional[str] = None,
    molecule_columns: List[str] = [],
    feature_columns: List[str] = [],
    train_frac: float = 0.1,
    also_train_on_partner: bool = True,
    validation_frac: float = 0.2,
    validation_ids: Optional[pd.Index] = None,
    module: Optional[pl.LightningModule] = None,
    is_log: bool = False,
    model: Callable = GAT(in_dim=3, hidden_dim=40, out_dim=1, num_heads=20),
    missing_substitute_value: float = 0.0,
    early_stopping: bool = True,
    max_epochs: int = 1000,
    inplace: bool = True,
    check_val_every_n_epoch: int = 1,
    predict_validation_ids: bool = False,
    silent: bool = False,
) -> Dataset:
    if validation_ids is None and predict_validation_ids:
        raise AttributeError("Cannot predict validation ids if validation_ids are not given")
    if result_column is None:
        result_column = column
    if not isinstance(result_column, (list, tuple)):
        result_column = [result_column]
    if not inplace:
        dataset = dataset.copy()

    gnnds = dataset.copy(columns=feature_columns)
    vals = dataset.values[molecule][column]
    if not is_log:
        vals = np.log(vals)
    mean = vals.mean()
    std = vals.std()
    gnnds.values[molecule]["gnninput"] = (vals - mean) / std
    molecule, mapping, partner_molecule = dataset.infer_mapping(molecule=molecule, mapping=mapping)
    if partner_column is None:
        partner_column = column
    vals = dataset.values[partner_molecule][partner_column]
    if not is_log:
        vals = np.log(vals)
    partner_mean = vals.mean()
    partner_std = vals.std()
    gnnds.values[partner_molecule]["gnninput"] = (vals - partner_mean) / partner_std

    if also_train_on_partner:
        train_molecules = [molecule, partner_molecule]
    else:
        train_molecules = [molecule]
    if validation_ids is None:
        train_mds, validation_mds = train_test_non_missing_no_overlap_iterable(
            dataset=gnnds,
            train_frac=train_frac,
            test_frac=validation_frac,
            molecule=train_molecules,
            non_missing_column="gnninput",
        )
    else:
        validation_mds = MaskedDataset.from_ids(dataset=gnnds, mask_ids={molecule: validation_ids})
        train_mds = non_missing_iterable(
            dataset=gnnds,
            molecule=train_molecules,
            column="gnninput",
            frac=train_frac,
            hidden_ids={molecule: validation_ids},
        )
    if silent:
        logger = None
    else:
        logger = ConsoleLogger()
    gnn_predictor = GnnPredictor(
        mapping=mapping,
        value_columns=feature_columns,
        molecule_columns=molecule_columns,
        target_column="gnninput",
        module=module,
        model=model,
        bidirectional_graph=True,
        missing_substitute_value=missing_substitute_value,
        logger=logger,
    )
    gnn_predictor.fit(
        train_mds=train_mds,
        test_mds=validation_mds,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        silent=silent,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )
    if not predict_validation_ids:
        predict_mds = mask_missing(dataset=gnnds, molecule=molecule, column="gnninput")
    else:
        predict_ids = gnnds.values[molecule]["gnninput"]
        if "sample" in validation_ids.names:
            predict_ids = predict_ids[predict_ids.isna() | predict_ids.index.isin(validation_ids)]
        else:
            predict_ids = predict_ids[
                predict_ids.isna() | predict_ids.index.get_level_values("id").isin(validation_ids)
            ]
        predict_mds = MaskedDataset.from_ids(dataset=gnnds, mask_ids={molecule: predict_ids.index})
    gnn_predictor.predict(
        mds=predict_mds, result_column=[f"res{i}" for i, c in enumerate(result_column)], silent=silent
    )
    for i, c in enumerate(result_column):
        vals = gnnds.values[molecule][f"res{i}"]
        if i == 0:
            vals = (vals * std) + mean
            if not is_log:
                vals = np.exp(vals)
            dataset.values[molecule][c] = vals
        else:
            vals = vals * std
            if not is_log:
                vals = np.exp(vals)
            dataset.values[molecule][c] = vals
    return dataset
