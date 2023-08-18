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
        super().__init__()
        self.out_dim = 1
        self.loss = loss.lower()
        if self.loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif self.loss == "nll":
            self.loss_fn = lambda y, target: F.gaussian_nll_loss(y[:, :, 0], target, torch.abs(y[:, :, 1]), eps=1e-4)
            self.out_dim = 2
        else:
            raise AttributeError("Loss has to be 'mse', or 'nll'!")
        self.model = GAT(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=self.out_dim, num_heads=num_heads)
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

    def forward(self, graph):
        features = graph.nodes["molecule"].data["x"].float()
        # molecule_type = graph.nodes['molecule'].data['type'].numpy()
        mask_nodes = graph.ndata["mask"]
        features = features.clone()
        if "hide" in graph.ndata:
            to_hide = graph.ndata["hide"]
            features[to_hide, 0] = self.hide_substitute_value
        features[mask_nodes, 0] = self.mask_substitute_value
        features[features.isnan()] = self.nan_substitute_value
        features_dict = {"molecule": features}
        # print(features_train)
        # Forward
        pred = self.model(graph, feat=features_dict)
        return pred

    def _log_metrics(self, y: torch.tensor, target: torch.tensor, loss: torch.tensor, prefix: str):
        batch_size = 1  # TODO
        if self.out_dim > 1:
            y = y[:, :, 0]
        mae = F.l1_loss(y, target).item()
        r2 = (torch.corrcoef(torch.t(torch.cat((y, target), 1)))[0, 1] ** 2).item()
        self.log(f"{prefix}_loss", loss, batch_size=batch_size)
        self.log(f"{prefix}_MAE", mae, batch_size=batch_size)
        self.log(f"{prefix}_r2", r2, batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        graph = batch
        # features = graph.nodes['molecule'].data['x'].float()
        # #molecule_type = graph.nodes['molecule'].data['type'].numpy()
        # target = graph.nodes['molecule'].data['target']
        # train_nodes = graph.ndata['mask']
        # features_train = features.clone()
        # if 'hide' in graph.ndata:
        #     to_hide = graph.ndata['hide']
        #     features_train[to_hide, 0] = self.hide_substitute_value
        # features_train[train_nodes, 0] = self.mask_substitute_value
        # features_train[features_train.isnan()] = self.nan_substitute_value
        # features_dict = {'molecule': features_train}
        # #print(features_train)
        # pred = self.model(graph, feat = features_dict
        # Forward)
        pred = self(graph)
        target = graph.nodes["molecule"].data["target"]
        mask_nodes = graph.nodes["molecule"].data["mask"]
        # Loss
        abundances_train = target[mask_nodes]
        abundances_train[abundances_train.isnan()] = self.nan_substitute_value
        train_loss = self.loss_fn(pred[mask_nodes], abundances_train)
        self._log_metrics(y=pred[mask_nodes], target=abundances_train, loss=train_loss, prefix='train')
        return train_loss

    def validation_step(self, batch, batch_idx):
        graph = batch
        # features = graph.nodes['molecule'].data['x'].float()
        # target = graph.nodes['molecule'].data['target']
        # features_test = features.clone()
        # if 'hide' in graph.nodes['molecule'].data:
        #     to_hide = graph.nodes['molecule'].data['hide']
        #     features_test[to_hide, 0] = self.hide_substitute_value
        # features_test[mask_nodes, 0] = self.mask_substitute_value
        # features_test[features_test.isnan()] = self.nan_substitute_value
        # features_dict = {'molecule': features_test}
        # pred = self.model(graph, feat = features_dict)
        pred = self(graph)
        target = graph.nodes["molecule"].data["target"]
        mask_nodes = graph.nodes["molecule"].data["mask"]
        abundances_test = target[mask_nodes]
        abundances_test[abundances_test.isnan()] = self.nan_substitute_value
        val_loss = self.loss_fn(pred[mask_nodes], abundances_test)
        self._log_metrics(y=pred[mask_nodes], target=abundances_test, loss=val_loss, prefix='validation')

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    # TODO: Deprectated, remove!
    def predict_masked_dataset(
        self,
        masked_dataset: AbstractMaskedDataset,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        result_column: str = "prediction",
        bidirectional_graph: bool = True,
    ):
        # graph = sample.molecule_set.create_graph(mapping=mapping).to_dgl()
        # sample.populate_graph_dgl(graph, value_columns=value_columns, mapping=mapping)
        graph_ds = GraphKeyDataset(
            masked_dataset=masked_dataset,
            mapping=mapping,
            value_columns=value_columns,
            molecule_columns=molecule_columns,
            target_column=target_column,
            bidirectional_graph=bidirectional_graph,
        )
        # we only have one dataset/graph so we only need to compute the node mapping once
        graph = graph_ds.graph
        dl = GraphDataLoader(graph_ds, batch_size=1)
        with torch.no_grad():
            trainer = Trainer()
            predictions = trainer.predict(self, dl)
            for i, prediction in enumerate(predictions):
                batch = graph_ds[i]
                mask_nodes = batch.ndata["mask"].detach().squeeze().numpy()
                assert np.all(graph.nodes.iloc[mask_nodes].type == graph.type_mapping[masked_dataset.molecule])
                prediction = prediction[mask_nodes]
                sample = graph_ds.index_to_sample(i)
                sample.values[masked_dataset.molecule].loc[:, result_column] = sample.dataset.missing_value
                molecules = graph.nodes.loc[mask_nodes.nonzero(), "molecule_id"].values  # type: ignore
                sample.values[masked_dataset.molecule].loc[molecules, result_column] = (
                    prediction.detach().squeeze().numpy()
                )

        # mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        # mask[mask_nodes] = 1
        # graph.nodes['molecule'].data['mask'] = mask
        # if self.hide_nodes is not None:
        #     hide = self.hide_nodes[ds_i]
        #     hide_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
        #     hide_mask[hide] = 1
        #     graph.nodes['molecule'].data['hide'] = hide_mask

    # def predict_masked_dataset(self, dataset: MaskedDataset, mapping: str = 'gene',
    #                             molecule_ids_to_predict: Optional[List] = None, value_columns: List[str] = ['abundance'],
    #                             molecule_columns: List[str] = [], target_column: str = 'abundance', bidirectional_graph: bool = True):
    #     #graph = sample.molecule_set.create_graph(mapping=mapping).to_dgl()
    #     #sample.populate_graph_dgl(graph, value_columns=value_columns, mapping=mapping)
    #     if molecule_ids_to_predict is None:
    #         missing_mask = []
    #         for column in value_columns:
    #             missing_mask.append(sample.missing_molecules(molecule=molecule, column=column).index)
    #         molecule_ids_to_predict = pd.concat(missing_mask).unique()
    #     graph = sample.dataset.create_graph(mapping=mapping, bidirectional=bidirectional_graph, cache=True)
    #     mask_nodes = graph.node_mapping[molecule].loc[molecule_ids_to_predict, 'node_id'].to_numpy()
    #     graph_ds = GraphDataSet(dataset_samples=[sample], mask_nodes=[mask_nodes], value_columns=value_columns,
    #                             molecule_columns=molecule_columns)
    #     return self(graph_ds[0])[mask_nodes].detach()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
