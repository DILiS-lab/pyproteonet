from typing import List, Union, Dict, Literal
from abc import abstractmethod

import lightning.pytorch as pl
from lightning.pytorch import Trainer
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from dgl.dataloading import GraphDataLoader

from ..data.abstract_masked_dataset import AbstractMaskedDataset
from ..dgl.masked_dataset_adapter import MaskedDatasetAdapter
from ..dgl.graph_key_dataset import GraphKeyDataset
from ..dgl.gnn_architectures.gat import GAT


class AbstractNodeImputer(pl.LightningModule):
    def __init__(
        self,
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.0001,
    ):
        super().__init__()
        self.nan_substitute_value = nan_substitute_value
        self.mask_substitute_value = mask_substitute_value
        self.hide_substitute_value = hide_substitute_value
        self.lr = lr
        self.save_hyperparameters(
            "nan_substitute_value",
            "mask_substitute_value",
            "hide_substitute_value",
            "lr",
        )

    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError()

    @abstractmethod
    def calculate_loss(self, pred, target):
        raise NotImplementedError

    @property
    def out_dim(self) -> int:
        return 1

    def forward(self, graph):
        target = graph.ndata["target"].float().clone()
        features = graph.ndata["features"].float()
        mask_nodes = graph.ndata["mask"]
        if "hidden" in graph.ndata:
            to_hide = graph.ndata["hidden"]
            #if len(to_hide.shape) == 1:
            #    to_hide = torch.unsqueeze(to_hide, dim=-1)
            target[to_hide] = self.hide_substitute_value
        #if len(mask_nodes.shape) == 1:
        #    mask_nodes = torch.unsqueeze(mask_nodes, dim=-1)
        target[mask_nodes] = self.mask_substitute_value
        target[target.isnan()] = self.nan_substitute_value
        assert features.isnan().sum() == 0
        #features_dict = {"molecule": torch.concat([target, features], axis=-1)}
        # Forward
        pred = self.model(graph, feat=torch.concat([target, features], axis=-1))
        #import pdb; pdb.set_trace()
        return pred

    def _log_metrics(self, y: torch.tensor, target: torch.tensor, loss: torch.tensor, prefix: str):
        batch_size = 1  # TODO
        if self.out_dim > 1:
            y = y[:, [0]]
        mae = F.l1_loss(y, target).item()
        mse = F.mse_loss(y, target).item()
        #pearson = (torch.corrcoef(torch.t(torch.cat((y, target), -1)))[0, 1]).item()
        y, target = y.squeeze(), target.squeeze() #TODO look why this is necessary when training on singe samples
        pearson = (torch.corrcoef(torch.t(torch.stack((y, target), -1)))[0, 1]).item()
        self.log(f"{prefix}_pearson", pearson, batch_size=batch_size)
        self.log(f"{prefix}_r2", pearson**2, batch_size=batch_size)
        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_mse", mse, batch_size=batch_size)
        self.log(f"{prefix}_rmse", mse**0.5, batch_size=batch_size)
        self.log(f"{prefix}_mae", mae, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        graph = batch
        pred = self(graph)
        target = graph.ndata["target"]
        mask_nodes = graph.ndata["mask"]
        # Loss
        abundances_train = target[mask_nodes]
        #abundances_train[abundances_train.isnan()] = self.nan_substitute_value
        assert abundances_train.isnan().sum().item() == 0
        train_loss = self.calculate_loss(pred[mask_nodes], abundances_train)
        self._log_metrics(y=pred[mask_nodes], target=abundances_train, loss=train_loss, prefix="train")
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        graph = batch
        pred = self(graph)
        target = graph.ndata["target"]
        mask_nodes = graph.ndata["mask"]
        abundances_test = target[mask_nodes]
        #abundances_test[abundances_test.isnan()] = self.nan_substitute_value
        assert abundances_test.isnan().sum().item() == 0
        val_loss = self.calculate_loss(pred[mask_nodes], abundances_test)
        self._log_metrics(y=pred[mask_nodes], target=abundances_test, loss=val_loss, prefix=f"validation")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        res = self(batch)
        return res

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
