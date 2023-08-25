from typing import Union, Dict, List, TYPE_CHECKING, Optional, Tuple, Iterable
from pathlib import Path
import logging
import collections

import numpy as np
from dgl.dataloading import GraphDataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import Logger
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ..dgl.gnn_architectures.gat import GAT
from ..dgl.gnn_architectures.resettable_module import ResettableModule
from ..lightning.node_regression_module import NodeRegressionModule
from ..lightning.console_logger import ConsoleLogger
from ..data.abstract_masked_dataset import AbstractMaskedDataset


class GnnPredictor:
    def __init__(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        module: Optional[pl.LightningModule] = None,
        model: ResettableModule = GAT(in_dim=4, hidden_dim=40, out_dim=1, num_heads=20),
        missing_substitute_value: float = 00,
        bidirectional_graph: bool = True,
        logger: Logger = ConsoleLogger(),  # type: ignore
    ):
        self.mapping = mapping
        self.value_columns = value_columns
        self.molecule_columns = molecule_columns
        self.target_column = target_column
        self.model = model
        self.missing_substitute_value = missing_substitute_value
        self.bidirectional_graph = bidirectional_graph
        self.module = module
        if self.module is None:
            if model is None:
                raise AttributeError("Please either specify a model or a lightning module!")
            self.module = NodeRegressionModule(
                model=self.model,
                nan_substitute_value=missing_substitute_value,
                mask_substitute_value=missing_substitute_value,
                hide_substitute_value=missing_substitute_value,
            )
        self.logger = logger
        self.trainer = None

    def fit(
        self,
        train_mds: AbstractMaskedDataset,
        test_mds: Optional[Union[AbstractMaskedDataset, Iterable[AbstractMaskedDataset]]],
        early_stopping: bool = True,
        max_epochs: int = 1000,
        silent: bool = False,
        check_val_every_n_epoch: int = 1,
        log_every_n_epochs = 1,
        continue_training: bool = False,
        eval_target_columns: Optional[Union[str, Iterable[str]]] = None,
        early_stopping_patience: int = 5,
        early_stopping_mds_index: Optional[int] = None,
    ):
        if eval_target_columns is None:
            eval_target_columns = [self.target_column]
        else:
            if not isinstance(eval_target_columns, collections.abc.Iterable):
                eval_target_columns = [eval_target_columns]
        if not continue_training:
            self.reset_parameters()
        train_gds = train_mds.get_graph_dataset_dgl(
            mapping=self.mapping,
            value_columns=self.value_columns,
            molecule_columns=self.molecule_columns,
            target_column=self.target_column,
            missing_column_value=self.missing_substitute_value,
        )
        train_dl = GraphDataLoader(train_gds, batch_size=1)  # TODO: think about batch size
        val_dl = None
        if test_mds is not None:
            val_dl = []
            if not isinstance(test_mds, (list, tuple)):
                test_mds = [test_mds]
            if len(test_mds) != len(eval_target_columns):
                raise AttributeError("The number of validation datasets has to equal to the number of validation target columns.")
            for mds, eval_target in zip(test_mds, eval_target_columns):
                test_gds = mds.get_graph_dataset_dgl(
                    mapping=self.mapping,
                    value_columns=self.value_columns,
                    molecule_columns=self.molecule_columns,
                    target_column=eval_target,
                    missing_column_value=self.missing_substitute_value,
                )
                val_dl.append(GraphDataLoader(test_gds, batch_size=1))
        #ptl_logger = logging.getLogger("lightning.pytorch")
        #plt_log_level = ptl_logger.level
        #if silent:
        #    ptl_logger.setLevel(logging.ERROR)
        callbacks = []
        if early_stopping:
            monitor = "validation_loss"
            if len(test_mds) > 1:
                if early_stopping_mds_index is None:
                    raise AttributeError("You have to specify the early_stopping_mds_index if specifying more than one evaluation masked dataset")
                monitor = f"validation_loss/dataloader_idx_{early_stopping_mds_index}"
            callbacks = [EarlyStopping(monitor=monitor, mode="min", patience=early_stopping_patience)]
        if not continue_training:
            self.trainer = Trainer(
                logger=self.logger,
                max_epochs=max_epochs,
                enable_checkpointing=False,
                enable_progress_bar=not silent,
                enable_model_summary=not silent,
                check_val_every_n_epoch=check_val_every_n_epoch,
                log_every_n_steps=len(train_gds) * log_every_n_epochs,
                callbacks=callbacks
            )
        else:
            if self.trainer is None:
                raise RuntimeError("You cannot specify continue_training without call fit with continue_training=False first")
            if self.trainer.fit_loop.epoch_progress.current.completed >= self.trainer.fit_loop.max_epochs:
                self.trainer.fit_loop.max_epochs += max_epochs
        self.trainer.fit(self.module, train_dataloaders=train_dl, val_dataloaders=val_dl)
        #if silent:
        #    ptl_logger.setLevel(plt_log_level)

    def predict(
        self,
        mds: AbstractMaskedDataset,
        result_column: Optional[Union[str, Tuple[str]]] = None,
        copy_non_predicted_from_target_column: bool = True,
        silent: bool = False,
    ):
        if result_column is None:
            result_column = self.target_column
        if not isinstance(result_column, (list, tuple)):
            result_column = [result_column]
        if self.module.out_dim != len(result_column):
            raise AttributeError(
                f"The prediction module has {self.module.out_dim} output dimensions but {len(result_column)} result column names were given!"
            )
        # graph = sample.molecule_set.create_graph(mapping=mapping).to_dgl()
        # sample.populate_graph_dgl(graph, value_columns=value_columns, mapping=mapping)
        graph_ds = mds.get_graph_dataset_dgl(
            mapping=self.mapping,
            value_columns=self.value_columns,
            molecule_columns=self.molecule_columns,
            target_column=self.target_column,
            missing_column_value=self.missing_substitute_value,
        )
        # we only have one dataset/graph so we only need to compute the node mapping once
        graph = graph_ds.graph
        dl = GraphDataLoader(graph_ds, batch_size=1)
        with torch.no_grad():
            #ptl_logger = logging.getLogger("lightning.pytorch")
            #plt_log_level = ptl_logger.level
            #if silent:
            #   ptl_logger.setLevel(logging.ERROR)
            #trainer = Trainer(enable_progress_bar=not silent, enable_model_summary=not silent)
            if self.trainer is None:
                raise RuntimeError("You need to call fit first.")
            predictions = self.trainer.predict(self.module, dl)
            #if silent:
            #    ptl_logger.setLevel(plt_log_level)
            for i, prediction in enumerate(predictions):
                batch = graph_ds[i]
                mask_nodes = batch.ndata["mask"].detach().squeeze().numpy()
                assert np.all(graph.nodes.iloc[mask_nodes].type == graph.type_mapping[mds.molecule])
                prediction = prediction[mask_nodes]
                sample = graph_ds.index_to_sample(i)
                if copy_non_predicted_from_target_column:
                    # TODO
                    sample.values[mds.molecule].loc[:, result_column[0]] = sample.values[mds.molecule].loc[
                        :, self.target_column
                    ]
                else:
                    sample.values[mds.molecule].loc[:, result_column] = sample.dataset.missing_value
                molecules = graph.nodes.loc[mask_nodes.nonzero(), "molecule_id"].values  # type: ignore
                prediction = prediction.detach().numpy()
                for i, c in enumerate(result_column):
                    sample.values[mds.molecule].loc[molecules, c] = prediction[:, i]

    @property
    def out_dim(self):
        return self.model.out_dim

    def reset_parameters(self):
        self.model.reset_parameters()

    def save_model(self, path: Union[str, Path]):
        path = Path(path)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
            },
            path,
        )

    def load_model(self, path: Union[str, Path]):
        path = Path(path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
