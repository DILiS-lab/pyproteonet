from typing import Union, Dict, List, TYPE_CHECKING, Optional
from pathlib import Path

from dgl.dataloading import GraphDataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
import torch
from torch import nn

from ..dgl.gnn_architectures.gat import GAT
from ..dgl.gnn_architectures.resettable_module import ResettableModule
from ..lightning.node_regression_module import NodeRegressionModule
from ..lightning.console_logger import ConsoleLogger
from ..data.masked_dataset import MaskedDataset


class GnnPredictor:
    def __init__(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        model: ResettableModule = GAT(in_dim=4, hidden_dim=40, out_dim=1, num_heads=20),
        bidirectional_graph: bool = True,
        logger: Logger = ConsoleLogger() # type: ignore
    ):
        self.mapping = mapping
        self.value_columns = value_columns
        self.molecule_columns = molecule_columns
        self.target_column = target_column
        self.model = model
        self.bidirectional_graph = bidirectional_graph
        self.module = NodeRegressionModule(model=self.model)
        self.logger = logger

    def fit(
        self,
        train_mds: MaskedDataset,
        test_mds: Optional["MaskedDataset"],
        max_epochs: int = 10,
        reset_parameters: bool = False,
    ):
        if reset_parameters:
            self.reset_parameters()
        train_gds = train_mds.get_graph_dataset_dgl(
            mapping=self.mapping,
            value_columns=self.value_columns,
            molecule_columns=self.molecule_columns,
            target_column=self.target_column,
        )
        train_dl = GraphDataLoader(train_gds, batch_size=1)
        val_dl = None
        if test_mds is not None:
            test_gds = test_mds.get_graph_dataset_dgl(
                mapping=self.mapping,
                value_columns=self.value_columns,
                molecule_columns=self.molecule_columns,
                target_column=self.target_column,
            )
            val_dl = GraphDataLoader(test_gds, batch_size=1)
        trainer = Trainer(logger=self.logger, max_epochs=max_epochs)
        trainer.fit(self.module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    def predict(self, mds: MaskedDataset, result_column: Optional[str] = None):
        if result_column is None:
            result_column = self.target_column
        self.module.predict_masked_dataset(
            mds,
            result_column=result_column,
            mapping=self.mapping,
            value_columns=self.value_columns,
            molecule_columns=self.molecule_columns,
            target_column=self.target_column,
        )

    def reset_parameters(self):
        self.model.reset_parameters()

    def save_model(self, path: Union[str, Path]):
        path = Path(path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            }, path)
        
    def load_model(self, path: Union[str, Path]):
        path = Path(path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])