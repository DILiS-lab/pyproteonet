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
from .abstract_node_imputer import AbstractNodeImputer
from ..dgl.gnn_architectures.deep_gat import DeepGAT


class GatNodeImputer(AbstractNodeImputer):
    def __init__(
        self,
        in_dim: int = 3,
        heads: int = [20,20],
        gat_dims: int = [40, 20],
        initial_dense_layers: List[int] = [],
        out_dim: int = 1,
        loss: Literal["mse", "gnll"] = "mse",
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.01,
        use_gatv2: bool = False,
    ):
        super().__init__(
                nan_substitute_value=nan_substitute_value,
                mask_substitute_value=mask_substitute_value,
                hide_substitute_value=hide_substitute_value,
                lr=lr,
            )
        self._out_dim = out_dim
        self.loss = loss.lower()
        if self.loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == "gnll":
            self.loss_fn = lambda y, target: F.gaussian_nll_loss(y[:, :out_dim], target, torch.abs(y[:, out_dim:]), eps=1e-1)
            self._out_dim = 2 * out_dim
        else:
            raise AttributeError("Loss has to be 'mse', or 'gnll'!")
        self._model = DeepGAT(in_dim=in_dim, heads=heads, gat_dims=gat_dims, out_dim=self.out_dim, use_gatv2=use_gatv2, initial_dense_layers=initial_dense_layers)

    @property
    def model(self):
        return self._model
    
    @property
    def out_dim(self):
        return self._out_dim
    
    def calculate_loss(self, pred, target):
        return self.loss_fn(pred, target)
