from typing import List, Literal, Optional
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GATv2Conv
import torch.nn.functional as F

from ....lightning.abstract_node_imputer import AbstractNodeImputer
from ....data.dataset import Dataset
from ....normalization.dnn_normalizer import DnnNormalizer


from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyproteonet.lightning import ConsoleLogger
from dgl.dataloading import GraphCollator

from ....data.dataset import Dataset
from ....data.masked_dataset import MaskedDataset
from ....masking.masked_dataset_generator import MaskedDatasetGenerator
from ....masking.train_eval import (
    train_eval_protein_and_mapped,
    train_eval_full_protein_and_mapped,
    train_eval_full_protein_and_mapped_backup,
    train_eval_full_molecule,
    train_eval_full_molecule_some_mapped
)
from ....dgl.collate import (
    masked_dataset_to_homogeneous_graph,
    masked_heterograph_to_homogeneous,
)
from ....lightning.uncertainty_gat_node_imputer import UncertaintyGatNodeImputer
from ....masking.missing_values import mask_missing

class UncertaintyGAT(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        heads: List[int],
        gat_dims: List[int],
        out_dim: int = 1,
        use_gatv2: bool = False,
        initial_dense_layers: List[int] = [],
        dropout: float = 0.0,
        num_embeddings: Optional[int] = None,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = None
        if num_embeddings is not None and embedding_dim is not None:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
            in_dim = embedding_dim + in_dim
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
        if self.embedding is not None:
            feat = torch.cat((self.embedding(graph.ndata[dgl.NID].int()), feat), dim=-1)
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
        dropout: float = 0.0,
        num_embeddings: Optional[int] = None,
        embedding_dim: Optional[int] = None,
    ):
        print(out_dim)
        super().__init__(
            nan_substitute_value=nan_substitute_value,
            mask_substitute_value=mask_substitute_value,
            hide_substitute_value=hide_substitute_value,
            lr=lr,
        )
        self._out_dim = out_dim
        print(num_embeddings, embedding_dim)
        self._model = UncertaintyGAT(
            in_dim=in_dim,
            heads=heads,
            gat_dims=gat_dims,
            out_dim=2 * out_dim,
            use_gatv2=use_gatv2,
            initial_dense_layers=initial_dense_layers,
            dropout=dropout,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
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
            return F.gaussian_nll_loss(pred[:, 0], target=target, var=torch.exp(pred[:, 1]))
        else:
            return F.mse_loss(pred[:, 0], target)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        res = self(batch)
        return res

def impute_all_sample_homogeneous_gnn(
    dataset: Dataset,
    molecule: str,
    mapping: str,
    column: str,
    partner_column: str,
    validation_fraction=0.1,
    training_fraction=0.3,
    partner_masking_fraction=0.5,
    result_column: Optional[str] = None,
    max_epochs: int = 10000,
    validation_frequency: int = 10,
    early_stopping_patience: int = 5,
    missing_substitute_value: float = -3,
    use_gatv2: bool = True,
    train_on_mapped: bool = True,
    uncertainty_column: Optional[str] = None,
    logger: Optional[object] = None,
    log_every_n_steps: int = 10,
    protein_gt_column: Optional[str] = None,
    max_partner_mnar_quantile: float = 0.0,
    embedding_dim: Optional[int] = None,
):
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )

    in_dataset = dataset.copy(
        columns={
            molecule: [column]
            if protein_gt_column is None
            else [column, protein_gt_column],
            partner_molecule: [partner_column],
        }
    )
    in_dataset.rename_values(
        columns={column: "abundance"},
        molecules=[molecule],
        inplace=True,
    )
    in_dataset.rename_values(
        columns={partner_column: "abundance"},
        molecules=[partner_molecule],
        inplace=True,
    )
    if protein_gt_column is not None:
        in_dataset.rename_values(
            columns={protein_gt_column: "abundance_gt"},
            molecules=[molecule],
            inplace=True,
        )
        in_dataset.values[partner_molecule]["abundance_gt"] = in_dataset.values[
            partner_molecule
        ]["abundance"]
        non_missing = in_dataset.values[molecule]["abundance"]
        gt = in_dataset.values[molecule]["abundance_gt"]
        mask = ~non_missing.isna()
        gt[mask] = non_missing[mask]
        in_dataset.values[molecule]["abundance_gt"] = gt
    normalizer = DnnNormalizer(columns=["abundance", "abundance_gt"], logarithmize=False)
    normalizer.normalize(dataset=in_dataset, inplace=True)

    # train_ds, eval_ds = train_eval_full_molecule_some_mapped(
    #     dataset=in_dataset,
    #     molecule=molecule,
    #     column="abundance",
    #     partner_column="abundance",
    #     mapping=mapping,
    #     validation_fraction=validation_fraction,
    #     training_fraction=training_fraction,
    #     partner_hide_fraction=partner_masking_fraction,
    # )
    # train_ds, eval_ds = train_eval_full_protein_and_mapped(
    #     dataset=in_dataset,
    #     molecule=molecule,
    #     column="abundance",
    #     partner_column="abundance",
    #     mapping=mapping,
    #     validation_fraction=validation_fraction,
    #     training_fraction=training_fraction,
    #     partner_masking_fraction=partner_masking_fraction,
    # )

    # determining the masking fraction to resemble the missing fraction of partner molecule in the dataset
    missing_mols = dataset.values[molecule][column]
    missing_mols = (
        missing_mols[missing_mols.isna()].index.get_level_values("id").unique()
    )
    mapped = dataset.get_mapped(
        molecule=molecule,
        partner_molecule=partner_molecule,
        mapping=mapping,
        partner_columns=[partner_column],
    )
    mapped_missing = mapped[mapped.index.get_level_values(molecule).isin(missing_mols)]
    missing_fraction = (mapped_missing.isna().sum() / mapped_missing.shape[0]).item()
    mapped_non_missing = mapped[
        ~mapped.index.get_level_values(molecule).isin(missing_mols)
    ]
    non_missing_fraction = (
        mapped_non_missing.isna().sum() / mapped_non_missing.shape[0]
    ).item()
    masking_fraction = missing_fraction - non_missing_fraction
    validation_ids = in_dataset.values[molecule]["abundance"]
    validation_ids = validation_ids[~validation_ids.isna()].sample(frac=0.1).index
    partner_validation_ids = in_dataset.values[partner_molecule]["abundance"]
    partner_validation_ids = (
        partner_validation_ids[~partner_validation_ids.isna()].sample(frac=0.1).index
    )
    validation_set = MaskedDataset.from_ids(
        dataset=in_dataset,
        mask_ids={molecule: validation_ids, partner_molecule: partner_validation_ids},
    )

    def masking_fn(in_ds):
        molecule_mask_ids = in_ds.values[molecule]["abundance"]
        molecule_mask_ids = molecule_mask_ids[~molecule_mask_ids.isna()]
        molecule_mask_ids = (
            molecule_mask_ids[~molecule_mask_ids.index.isin(validation_ids)]
            .sample(frac=0.1)
            .index
        )
        partner_mask_ids = in_ds.values[partner_molecule]["abundance"]
        partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
        partner_mask_ids = (
            partner_mask_ids[~partner_mask_ids.index.isin(partner_validation_ids)]
            .sample(frac=masking_fraction)
            .index
        )
        return MaskedDataset.from_ids(
            dataset=in_ds,
            mask_ids={molecule: molecule_mask_ids, partner_molecule: partner_mask_ids},
            hidden_ids={molecule: validation_ids},
        )
    mask_ds = MaskedDatasetGenerator(datasets=[in_dataset], generator_fn=masking_fn)



    collator = GraphCollator()
    collate_fn = lambda masked_datasets: collator.collate(
        masked_dataset_to_homogeneous_graph(
            masked_datasets=masked_datasets,
            mappings=[mapping],
            target="abundance",
            features=[],
        )
    )
    train_dl = DataLoader(mask_ds, batch_size=1, collate_fn=collate_fn)
    graph = list(train_dl)[0]
    num_embeddings = int(graph.ndata[dgl.NID].max().item() + 1)
    eval_dl = DataLoader([validation_set], batch_size=1, collate_fn=collate_fn)
    early_stopping_monitor = "validation_loss"
    if protein_gt_column is not None:
        gt_collate_fn = lambda masked_datasets: collator.collate(
            masked_dataset_to_homogeneous_graph(
                masked_datasets=masked_datasets,
                mappings=[mapping],
                target="abundance_gt",
                features=[],
            )
        )
        gt_ds = mask_missing(dataset=in_dataset, molecule=molecule, column="abundance")
        gt_dl = DataLoader([gt_ds], batch_size=1, collate_fn=gt_collate_fn)
        eval_dl = [eval_dl, gt_dl]
        early_stopping_monitor = "validation_loss/dataloader_idx_0"

    num_samples = len(in_dataset.sample_names)
    # heads = [num_samples, num_samples]  # [num_samples]
    # dimensions = [8 * num_smples, 4*num_samples, 2*num_samples]
    heads = [4 * num_samples, 4 * num_samples, 4 * num_samples]
    #heads = [8 * num_samples]
    #dimensions = [8]
    dimensions = [8, 8, 4] # [1]#, num_samples, num_samples]
    print(heads)
    print(dimensions)
    # module = GatNodeImputer(in_dim = num_samples + 2,
    #                         heads=heads, gat_dims=dimensions,
    #                         mask_substitute_value=missing_substitute_value, hide_substitute_value=missing_substitute_value,
    #                         nan_substitute_value=missing_substitute_value,
    #                         out_dim=num_samples, use_gatv2=use_gatv2, initial_dense_layers=[8*num_samples, num_samples])
    module = UncertaintyGatNodeImputer(
        in_dim=num_samples + 2,
        heads=heads,
        gat_dims=dimensions,
        mask_substitute_value=missing_substitute_value,
        hide_substitute_value=missing_substitute_value,
        nan_substitute_value=missing_substitute_value,
        out_dim=num_samples,
        use_gatv2=use_gatv2,
        initial_dense_layers=[
            8 * num_samples,
            2 * num_samples
        ],  # [8 * num_samples, 2 * num_samples]
        dropout=0.2,
        lr=0.001,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim
    )

    if logger is None:
        logger = ConsoleLogger()
    module.uncertainty_loss = False
    if uncertainty_column is None:
        trainer = Trainer(
            logger=logger,
            max_epochs=max_epochs,
            enable_checkpointing=False,
            check_val_every_n_epoch=validation_frequency,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    mode="min",
                    patience=early_stopping_patience,
                )
            ],
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    else:
        module.uncertainty_loss = True
        module.lr = 0.0001
        trainer = Trainer(
            logger=logger,
            max_epochs=max_epochs,
            enable_checkpointing=False,
            check_val_every_n_epoch=validation_frequency,
            log_every_n_steps=log_every_n_steps,
            callbacks=[
                EarlyStopping(
                    monitor=early_stopping_monitor,
                    mode="min",
                    patience=early_stopping_patience,
                )
            ],
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=eval_dl)

    predict_ds = mask_missing(dataset=in_dataset, molecule=molecule, column="abundance")
    predict_graph = predict_ds.to_dgl_graph(
        molecule_features={
            mol: ["abundance"] for mol in predict_ds.dataset.molecules.keys()
        },
        mappings=[mapping],
    )
    ntypes = predict_graph.ntypes
    predict_graph = masked_heterograph_to_homogeneous(
        masked_heterographs=[predict_graph], target="abundance", features=[]
    )[0]
    protein_index = ntypes.index(molecule)
    protein_mask = (predict_graph.ndata[dgl.NTYPE] == protein_index).type(torch.bool)
    res = trainer.predict(
        module, DataLoader([predict_graph], batch_size=1, collate_fn=collator.collate)
    )[0]
    prot_res = res[protein_mask]
    masked_proteins = predict_graph.ndata["mask"][protein_mask].type(torch.bool)
    protein_ids = predict_graph.ndata[dgl.NID][protein_mask].type(torch.int64)
    prot_res = prot_res[protein_ids, :]
    masked_proteins = masked_proteins[protein_ids, :]
    mat_pd = in_dataset.get_samples_value_matrix(molecule=molecule, column="abundance")
    mat = mat_pd.to_numpy()
    mat[masked_proteins.numpy()] = prot_res[masked_proteins].numpy()[:, 0]
    # mat[:, :] = prot_res[:, :].numpy()
    mat_pd.loc[:, :] = mat
    in_dataset.set_samples_value_matrix(
        matrix=mat_pd, molecule=molecule, column="abundance"
    )
    normalizer.unnormalize(dataset=in_dataset, inplace=True)
    vals = dataset.values[molecule][column]
    res_vals = in_dataset.values[molecule]["abundance"]
    vals.loc[vals.isna(), :] = res_vals.loc[vals.isna(), :]
    if result_column is not None:
        dataset.values[molecule][result_column] = vals
    if uncertainty_column is not None:
        mat[:, :] = np.nan
        mat[masked_proteins.numpy()] = prot_res[masked_proteins].numpy()[:, 1]
        mat_pd.loc[:, :] = mat
        dataset.set_samples_value_matrix(
            matrix=mat_pd, molecule=molecule, column=uncertainty_column
        )
    return vals
