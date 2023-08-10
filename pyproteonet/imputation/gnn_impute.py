from typing import Optional, Callable

import numpy as np

from ..data.dataset import Dataset
from ..predictors.gnn_predictor import GnnPredictor
from ..processing.masking import train_test_non_missing_no_overlap_iterable, mask_missing
from ..dgl.gnn_architectures.gat import GAT
from ..lightning.console_logger import ConsoleLogger


def gnn_impute(
    dataset: Dataset,
    molecule: str = "peptide",
    column: str = "normalized",
    mapping: str = "protein_group",
    result_column: str = "gnnimp",
    partner_molecule: Optional[str] = None,
    partner_column: Optional[str] = None,
    train_frac: float = 0.1,
    test_frac: float = 0.2,
    model: Callable = GAT(in_dim=3, hidden_dim=40, out_dim=1, num_heads=20),
    missing_substitute_value: float = 0.0,
    max_epochs: int = 20,
    inplace: bool = True,
) -> Dataset:
    if result_column is None:
        result_column = column
    if not inplace:
        dataset = dataset.copy()

    gnnds = dataset.copy(columns=[])
    vals = np.log(dataset.values[molecule][column])
    mean = vals.mean()
    std = vals.std()
    #import pdb; pdb.set_trace()
    gnnds.values[molecule]["gnninput"] = (vals - mean) / std
    if partner_molecule is not None:
        if partner_column is None:
            partner_column = column
        vals = np.log(dataset.values[partner_molecule][partner_column])
        partner_mean = vals.mean()
        partner_std = vals.std()
        gnnds.values[partner_molecule]["gnninput"] = (vals - partner_mean) / partner_std

    train_mds, test_mds = train_test_non_missing_no_overlap_iterable(
        dataset=gnnds, train_frac=train_frac, test_frac=test_frac, molecule=molecule, column="gnninput"
    )
    gnn_predictor = GnnPredictor(
        mapping=mapping,
        value_columns=["gnninput"],
        molecule_columns=[],
        target_column="gnninput",
        model=model,
        bidirectional_graph=True,
        missing_substitute_value=missing_substitute_value,
        logger=ConsoleLogger(),
    )
    gnn_predictor.fit(train_mds=train_mds, test_mds=test_mds, max_epochs=max_epochs)
    missing_mds = mask_missing(dataset=gnnds, molecule=molecule, column="gnninput")
    gnn_predictor.predict(mds=missing_mds, result_column="gnnresult")
    vals = gnnds.values[molecule]["gnnresult"]
    dataset.values[molecule][result_column] = np.exp((vals * std) + mean)
    return dataset
