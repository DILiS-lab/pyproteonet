from typing import List, Union, Dict, Literal

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


class GatNodeRegressor(pl.LightningModule):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 40,
        num_heads: int = 20,
        loss: Literal["mse", "nll"] = "mse",
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.0001,
    ):
        super().__init__(
                nan_substitute_value=nan_substitute_value,
                mask_substitute_value=mask_substitute_value,
                hide_substitute_value=hide_substitute_value,
                lr=lr,
            )
        self._out_dim = 1
        self.loss = loss.lower()
        if self.loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == "nll":
            self.loss_fn = lambda y, target: F.gaussian_nll_loss(y[:, :, 0], target, torch.abs(y[:, :, 1]), eps=1e-4)
            self._out_dim = 2
        else:
            raise AttributeError("Loss has to be 'mse', or 'nll'!")
        self._model = GAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=self.out_dim, num_heads=num_heads)

    @property
    def model(self):
        return self._model
    
    @property
    def out_dim(self):
        return self._out_dim
    
    def calculate_loss(self, pred, target):
        return self.loss_fn(pred, target)
