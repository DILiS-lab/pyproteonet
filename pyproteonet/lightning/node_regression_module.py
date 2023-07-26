from typing import List, Union, Dict

import lightning.pytorch as pl
from lightning.pytorch import Trainer
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from dgl.dataloading import GraphDataLoader

from ..data.masked_dataset import MaskedDataset
from ..dgl.graph_data_set import GraphDataSet


class NodeRegressionModule(pl.LightningModule):
    def __init__(
        self,
        model,
        nan_substitute_value: float = 0.0,
        mask_substitute_value: float = 0.0,
        hide_substitute_value: float = 0.0,
        lr: float = 0.02,
    ):
        super().__init__()
        self.model = model
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

    def training_step(self, batch, batch_idx):
        graph = batch
        batch_size = 1  # TODO
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
        train_loss = F.mse_loss(pred[mask_nodes], abundances_train)
        train_MAE = F.l1_loss(pred[mask_nodes], abundances_train).item()
        train_r2 = (torch.corrcoef(torch.t(torch.cat((pred[mask_nodes], abundances_train), 1)))[0, 1] ** 2).item()
        self.log("train_loss", train_loss, batch_size=batch_size)
        self.log("train_MAE", train_MAE, batch_size=batch_size)
        self.log("train_r2", train_r2, batch_size=batch_size)
        return train_loss

    def validation_step(self, batch, batch_idx):
        graph = batch
        batch_size = 1  # TODO
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
        val_loss = F.mse_loss(pred[mask_nodes], abundances_test)
        val_MAE = F.l1_loss(pred[mask_nodes], abundances_test).item()
        val_r2 = (torch.corrcoef(torch.t(torch.cat((pred[mask_nodes], abundances_test), 1)))[0, 1] ** 2).item()
        self.log("val_loss", val_loss, batch_size=batch_size)
        self.log("val_MAE", val_MAE, batch_size=batch_size)
        self.log("val_r2", val_r2, batch_size=batch_size)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def predict_masked_dataset(
        self,
        masked_dataset: MaskedDataset,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        result_column: str = 'prediction',
        bidirectional_graph: bool = True,
    ):
        # graph = sample.molecule_set.create_graph(mapping=mapping).to_dgl()
        # sample.populate_graph_dgl(graph, value_columns=value_columns, mapping=mapping)
        graph_ds = GraphDataSet(masked_datasets=[masked_dataset], mapping=mapping, value_columns=value_columns, 
                                molecule_columns=molecule_columns, target_column=target_column, bidirectional_graph=bidirectional_graph)
        #we only have one dataset/graph so we only need to compute the node mapping once
        graph = graph_ds.index_to_molecule_graph(0)
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
                molecules = graph.nodes.loc[mask_nodes.nonzero(), 'molecule_id'].values # type: ignore
                sample.values[masked_dataset.molecule].loc[molecules, result_column] = prediction.detach().squeeze().numpy()

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
