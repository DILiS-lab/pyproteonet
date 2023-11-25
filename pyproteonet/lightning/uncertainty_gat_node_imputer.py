from typing import List, Literal

import torch
import torch.nn.functional as F
from .abstract_node_imputer import AbstractNodeImputer

from typing import List

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GATv2Conv
import torch.nn.functional as F
import matplotlib.pyplot as plt


class UncertaintyGAT(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        heads: List[int],
        gat_dims: List[int],
        out_dim: int = 1,
        use_gatv2: bool = False,
        initial_dense_layers: List[int] = [],
        dropout: float = 0.0
    ):
        super().__init__()
        if len(initial_dense_layers) > 0:
            dense_layers = []
            for dim in initial_dense_layers:
                dense_layers.append(nn.Linear(in_dim, dim))
                dense_layers.append(nn.Dropout(p=dropout))
                dense_layers.append(nn.ReLU())
                in_dim = dim
            self.initial_layers = nn.Sequential(*dense_layers)
        else:
            self.initial_layers = nn.Identity()
        layers = []
        assert len(gat_dims) == len(heads)
        layer_type = GATConv
        if use_gatv2:
            layer_type = GATv2Conv
        for i, (d, h) in enumerate(zip(gat_dims, heads)):
            if i == 0:
                last_d = in_dim
                last_h = 1
            else:
                last_d = gat_dims[i - 1]
                last_h = heads[i - 1]
            layers.append(
                layer_type(in_feats=last_d * last_h, out_feats=d, num_heads=h, feat_drop=dropout, attn_drop=dropout)
            )
        self.gat_layers: List[layer_type] = nn.ModuleList(layers)
        self.out_layer = layer_type(
            in_feats=gat_dims[-1] * heads[-1], out_feats=out_dim, num_heads=1
        )

    def reshape_multihead_output(self, h):
        h_concat = []
        for h_idx in range(h.size()[1]):
            h_concat.append(h[:, h_idx])
        h = torch.cat(h_concat, axis=-1)
        return h

    def forward(self, graph, feat, eweight=None):
        # graph = dgl.to_homogeneous(graph, ndata = ['x'])
        # feat = feat['molecule']
        feat = self.initial_layers(feat)
        for layer in self.gat_layers:
            feat = layer(graph, feat)
            feat = self.reshape_multihead_output(F.relu(feat))
        feat = self.out_layer(graph, feat)
        feat = torch.squeeze(feat, dim=1)
        return feat

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()


class UncertaintyGatNodeImputer(AbstractNodeImputer):
    def __init__(
        self,
        in_dim: int = 3,
        heads: int = [20, 20],
        gat_dims: int = [40, 20],
        out_dim: int = 1,
        initial_dense_layers: List[int] = [],
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.0001,
        use_gatv2: bool = False,
        uncertainty_loss: bool = True,
        dropout: float = 0.0
    ):
        print(out_dim)
        super().__init__(
            nan_substitute_value=nan_substitute_value,
            mask_substitute_value=mask_substitute_value,
            hide_substitute_value=hide_substitute_value,
            lr=lr,
        )
        self._out_dim = out_dim
        self._model = UncertaintyGAT(
            in_dim=in_dim,
            heads=heads,
            gat_dims=gat_dims,
            out_dim=2 * out_dim,
            use_gatv2=use_gatv2,
            initial_dense_layers=initial_dense_layers,
            dropout=dropout
        )
        self.uncertainty_loss = uncertainty_loss

    def _log_metrics(
        self, y: torch.tensor, target: torch.tensor, loss: torch.tensor, prefix: str
    ):
        batch_size = 1  # TODO
        uncertainty = y[:, 1]
        y = y[:, 0]
        # if self.out_dim > 1:
        #    y = y[:, :, :self.num_abundance_features]
        mae = F.l1_loss(y, target).item()
        mse = F.mse_loss(y, target).item()
        # pearson = (torch.corrcoef(torch.t(torch.cat((y, target), -1)))[0, 1]).item()
        y, target = (
            y.squeeze(),
            target.squeeze(),
        )  # TODO look why this is necessary when training on singe samples
        pearson = (torch.corrcoef(torch.t(torch.stack((y, target), -1)))[0, 1]).item()
        self.log(f"{prefix}_pearson", pearson, batch_size=batch_size)
        self.log(f"{prefix}_r2", pearson**2, batch_size=batch_size)
        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_mse", mse, batch_size=batch_size)
        self.log(f"{prefix}_rmse", mse**0.5, batch_size=batch_size)
        self.log(f"{prefix}_mae", mae, batch_size=batch_size)
        uncertainty_pearson = (torch.corrcoef(torch.t(torch.stack((y, uncertainty), -1)))[0, 1]).item()
        self.log(f"{prefix}_uncertainty_pearson", uncertainty_pearson, batch_size=batch_size)

    @property
    def model(self):
        return self._model

    def forward(self, graph):
        pred = super().forward(graph)
        pred = pred.reshape(-1, self.out_dim, 2)
        return pred

    @property
    def out_dim(self):
        return self._out_dim

    def calculate_loss(self, pred, target):
        if self.uncertainty_loss:
            return F.gaussian_nll_loss(pred[:, 0], target=target, var=torch.abs(pred[:, 1]), eps=1e-1)
        else:
            return F.mse_loss(pred[:, 0], target)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        res = self(batch)
        return res
