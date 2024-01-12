from typing import List, Literal, Optional, Tuple
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GATv2Conv

from ....lightning.abstract_node_imputer import AbstractNodeImputer
from ....lightning.training_early_stopping import TrainingEarlyStopping
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
from ....dgl.collate import (
    masked_dataset_to_homogeneous_graph,
    masked_heterograph_to_homogeneous,
)
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
        average_heads: bool = False,
    ):
        super().__init__()
        self.embedding = None
        self.average_heads = average_heads
        if num_embeddings is not None and embedding_dim is not None:
            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=embedding_dim
            )
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
                layer_type(
                    in_feats=last_d if self.average_heads else last_d * last_h,
                    out_feats=d,
                    num_heads=h,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            )
        self.gat_layers: List[layer_type] = nn.ModuleList(layers)
        self.out_layer = layer_type(
            in_feats=gat_dims[-1] if self.average_heads else gat_dims[-1] * heads[-1], out_feats=out_dim, num_heads=1
        )

    def reshape_multihead_output(self, h):
        if self.average_heads:
            return h.mean(dim=-2)
        else:
            h_concat = []
            for h_idx in range(h.size()[1]):
                h_concat.append(h[:, h_idx])
            return torch.cat(h_concat, axis=-1)

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
        average_heads: bool = False
    ):
        #print(out_dim)
        super().__init__(
            nan_substitute_value=nan_substitute_value,
            mask_substitute_value=mask_substitute_value,
            hide_substitute_value=hide_substitute_value,
            lr=lr,
        )
        self._out_dim = out_dim
        #print(num_embeddings, embedding_dim)
        self._model = UncertaintyGAT(
            in_dim=in_dim,
            heads=heads,
            gat_dims=gat_dims,
            out_dim=2 * out_dim,
            use_gatv2=use_gatv2,
            initial_dense_layers=initial_dense_layers,
            dropout=dropout,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            average_heads=average_heads,
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
        uncertainty_pearson = (
            torch.corrcoef(torch.t(torch.stack((y, uncertainty), -1)))[0, 1]
        ).item()
        self.log(
            f"{prefix}_uncertainty_pearson", uncertainty_pearson, batch_size=batch_size
        )

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
            return F.gaussian_nll_loss(
                pred[:, 0], target=target, var=torch.exp(pred[:, 1])
            )
        else:
            return F.mse_loss(pred[:, 0], target)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        res = self(batch)
        return res


def impute_homogeneous_gnn(
    dataset: Dataset,
    molecule: str,
    mapping: str,
    column: str,
    partner_column: str,
    training_fraction=0.25,
    feature_columns:  Optional[List[str]] = None,
    partner_feature_columns: Optional[List[str]] = None,
    result_column: Optional[str] = None,
    partner_result_column: Optional[str] = None,
    max_epochs: int = 10000,
    validation_frequency: Optional[int] = None,
    early_stopping_patience: int = 7,
    missing_substitute_value: float = -3,
    use_gatv2: bool = True,
    mask_partner: bool = True,
    train_on_partner: bool = True,
    uncertainty_column: Optional[str] = None,
    logger: Optional[object] = None,
    log_every_n_steps: int = 30,
    embedding_dim: Optional[int] = None,
    train_sample_wise: bool = False,
    molecule_gt_column: Optional[str] = None,
    epoch_size: int = 1
)->pd.Series:
    """Impute missing values using a homogenous graph neural network applied on the molecule graph created from two molecule types like proteins and their assigned peptides.

    Args:
        dataset (Dataset): The dataset to impute.
        molecule (str): The main molecule type to impute (e.g. "protein").
        column (str): The value column of the main molecule type to impute (e.g. "abundance").
        mapping (str): The name of the mapping, connecting the main molecule type with a partner molecule type (e.g. "protein-peptide").
        partner_column (str): The value column of the partner molecule type to impute.
        training_fraction (float, optional): Mean fraction of molecules masked during training (The masking fraction for every epoch is randomly drawn from the (0.5 * training_fraction, 1.5 * training_fraction) interval). Defaults to 0.25.
        feature_columns (Optional[List[str]], optional): Names of additional value columns to use as featues for the main molecule. Defaults to None.
        partner_feature_columns (Optional[List[str]], optional): Names of additional value columns to use as featues for the partner molecule (should be the same number as for the main molecule to allow creation of a homogeneous graph). Defaults to None.
        result_column (Optional[str], optional): If given, imputed results for the main molecule will be stored unders this name. Defaults to None.
        partner_result_column (Optional[str], optional): If given, imputed results for the partner molecule will be stored under this name. Defaults to None.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10000.
        validation_frequency (Optional[int], optional): If given validation is run every validation_frequency epochs. Defaults to None.
        early_stopping_patience (int, optional): Number of epochs after which the training is stopped if the training loss does not improve. Defaults to 7.
        missing_substitute_value (float, optional): Value to replace missing or masked values with. Defaults to -3.
        use_gatv2 (bool, optional): Whether to use the DGL GATv2 graph attention layers or the original DLG GAT layers. Defaults to True.
        mask_partner (bool, optional): Whether to randomly mask both the main and partner molecule during training or only the main molecules. Defaults to True.
        train_on_partner (bool, optional): Whether to compute training loss on masked main and partner molecules or only on the main molecules. Defaults to True.
        uncertainty_column (Optional[str], optional): Whether to predict an uncertainty value. Defaults to None.
        logger (Optional[object], optional): If given this logger is used for logger (should have the lightning logger interface). Defaults to None.
        log_every_n_steps (int, optional): How often to log. Defaults to 30.
        embedding_dim (Optional[int], optional): If given every molecule will have a trainable embedding of this dimension. Defaults to None.
        train_sample_wise (bool, optional): Whether a training step operates only on a single sample or the whole dataset. Defaults to False.
        molecule_gt_column (Optional[str], optional): If given some metrics comparing predictions with ground truth values will be logged during training (helpful to evaluate training progress with respect to a ground truth). Defaults to None.
        epoch_size (int, optional): Number of training runs on the dataset that make up an epoch. Defaults to 1.

    Raises:
        ValueError: Raised when main and partner molecule feature columns are not of the same length.

    Returns:
        pd.Series: The imputed values for the main molecule.
    """
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    if feature_columns is None:
        feature_columns = []
    if partner_feature_columns is None:
        partner_feature_columns = feature_columns
    if len(feature_columns) != len(partner_feature_columns):
        raise ValueError(
            "The number of molecule and partner molecule feature columns must be the same to allow combining them into a homogeneous graph."
        )

    in_dataset = dataset.copy(
        columns={
            molecule: [column]
            if molecule_gt_column is None
            else [column, molecule_gt_column],
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
    if molecule_gt_column is not None:
        in_dataset.rename_values(
            columns={molecule_gt_column: "abundance_gt"},
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
        in_dataset.values[partner_molecule]["abundance_gt"] = in_dataset.values[partner_molecule]["abundance"]
    for i, col in enumerate(feature_columns):
        in_dataset.values[molecule][f'f{i}'] = dataset.values[molecule][col]
    for i, col in enumerate(partner_feature_columns):
        in_dataset.values[partner_molecule][f'f{i}'] = dataset.values[partner_molecule][col]
    feature_names = [f"f{i}" for i in range(len(feature_columns))]
    normalizer = DnnNormalizer(
        columns=["abundance", "abundance_gt"] + feature_names, logarithmize=False
    )
    normalizer.normalize(dataset=in_dataset, inplace=True)

    # # determining the masking fraction to resemble the missing fraction of partner molecule in the dataset
    # missing_mols = dataset.values[molecule][column]
    # missing_mols = (
    #     missing_mols[missing_mols.isna()].index.get_level_values("id").unique()
    # )
    # mapped = dataset.get_mapped(
    #     molecule=molecule,
    #     partner_molecule=partner_molecule,
    #     mapping=mapping,
    #     partner_columns=[partner_column],
    # )
    # mapped_missing = mapped[mapped.index.get_level_values(molecule).isin(missing_mols)]
    # missing_fraction = (mapped_missing.isna().sum() / mapped_missing.shape[0]).item()
    # overall_missing_fraction = dataset.values[partner_molecule][partner_column]
    # overall_missing_fraction = overall_missing_fraction.isna().sum() / overall_missing_fraction.shape[0]
    # # mapped_non_missing = mapped[
    # #     ~mapped.index.get_level_values(molecule).isin(missing_mols)
    # # ]
    # # non_missing_fraction = (
    # #     mapped_non_missing.isna().sum() / mapped_non_missing.shape[0]
    # # ).item()
    # # masking_fraction = missing_fraction - non_missing_fraction
    # masking_fraction = missing_fraction - overall_missing_fraction
    # partner_vals = dataset.values[partner_molecule][partner_column]
    # masking_fraction = masking_fraction / (1 - partner_vals.isna().sum() / partner_vals.shape[0])
    # assert masking_fraction > 0

    validation_ids = in_dataset.values[molecule]["abundance"]
    validation_ids = validation_ids[~validation_ids.isna()].sample(frac=0.2).index
    # partner_validation_ids = in_dataset.values[partner_molecule]["abundance"]
    # partner_validation_ids = (
    #     partner_validation_ids[~partner_validation_ids.isna()].sample(frac=0.1).index
    # )
    validation_set = MaskedDataset.from_ids(
        dataset=in_dataset,
        mask_ids={
            molecule: validation_ids
        },  # , partner_molecule: partner_validation_ids},
    )
    if train_sample_wise:
        validation_set = [(validation_set, [s]) for s in in_dataset.sample_names]
    else:
        validation_set = [(validation_set, None)]

    mapping_df = dataset.mappings[mapping].df

    rng = np.random.default_rng()
    def masking_fn(in_ds):
        mask_ids = {}
        hidden_ids = {}#molecule: validation_ids}
        epoch_masking_fraction = rng.uniform(0.5 * training_fraction, 1.5 * training_fraction)
        molecule_ids = in_ds.molecules[molecule].sample(frac=epoch_masking_fraction).index
        molecule_mask_ids = in_ds.values[molecule]["abundance"]
        molecule_mask_ids = molecule_mask_ids[molecule_mask_ids.index.get_level_values("id").isin(molecule_ids)]
        molecule_mask_ids = molecule_mask_ids[~molecule_mask_ids.isna()].index
        # molecule_mask_ids = (
        #     molecule_mask_ids#[~molecule_mask_ids.index.isin(validation_ids)]
        #     .sample(frac=training_fraction)
        #     .index
        # )
        mask_ids[molecule] = molecule_mask_ids
        molecules = molecule_mask_ids.get_level_values("id").unique()
        # partner_molecules = (
        #     mapping_df[mapping_df.index.get_level_values(molecule).isin(molecules)]
        #     .index.get_level_values(partner_molecule)
        #     .unique()
        # )
        # partner_mask_ids = in_ds.values[partner_molecule]["abundance"]
        # partner_mask_ids = partner_mask_ids[
        #     partner_mask_ids.index.get_level_values("id").isin(partner_molecules)
        # ]
        # partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
        # partner_mask_ids = partner_mask_ids.sample(  # [~partner_mask_ids.index.isin(partner_validation_ids)]
        #     frac=masking_fraction
        # ).index
        partner_mask_ids = in_ds.values[partner_molecule]["abundance"]
        partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
        partner_mask_ids = partner_mask_ids.sample(frac=epoch_masking_fraction).index
        if train_on_partner:
            mask_ids[partner_molecule] = partner_mask_ids
        else:
            if mask_partner:
                hidden_ids[partner_molecule] = partner_mask_ids
        return MaskedDataset.from_ids(
            dataset=in_ds,
            mask_ids=mask_ids,
            hidden_ids=hidden_ids,
        )

    mask_ds = MaskedDatasetGenerator(datasets=[in_dataset], generator_fn=masking_fn, sample_wise=train_sample_wise, epoch_size_multiplier=epoch_size)

    collator = GraphCollator()

    def collate_fn(masked_ds_samples: Tuple[MaskedDataset, List[str]]):
        masked_datasets = []
        samples_lists = []
        for masked_ds, samples in masked_ds_samples:
            masked_datasets.append(masked_ds)
            samples_lists.append(samples)
        return collator.collate(
            masked_dataset_to_homogeneous_graph(
                masked_datasets=masked_datasets,
                mappings=[mapping],
                target="abundance",
                features=feature_names,
                sample_lists=samples_lists,
            )
        )

    def gt_collate_fn(masked_ds_samples: Tuple[MaskedDataset, List[str]]):
        masked_datasets = []
        samples_lists = []
        for masked_ds, samples in masked_ds_samples:
            masked_datasets.append(masked_ds)
            samples_lists.append(samples)
        return collator.collate(
            masked_dataset_to_homogeneous_graph(
                masked_datasets=masked_datasets,
                mappings=[mapping],
                target="abundance_gt",
                features=feature_names,
                sample_lists=samples_lists,
            )
        )

    train_dl = DataLoader(mask_ds, batch_size=1, collate_fn=collate_fn)
    graph = list(train_dl)[0]
    num_embeddings = int(graph.ndata[dgl.NID].max().item() + 1)
    if validation_frequency is not None:
        val_dls = [DataLoader(validation_set, batch_size=1, collate_fn=collate_fn)]
    else:
        val_dls = []
    early_stopping_monitor = "train_loss"
    if molecule_gt_column is not None:
        gt_ds = mask_missing(dataset=in_dataset, molecule_columns={molecule:"abundance"})
        if train_sample_wise:
            gt_ds = [(gt_ds, [s]) for s in in_dataset.sample_names]
        else:
            gt_ds = [(gt_ds, None)]
        gt_dl = DataLoader(gt_ds, batch_size=1, collate_fn=gt_collate_fn)
        val_dls.append(gt_dl)
        #early_stopping_monitor = "validation_loss/dataloader_idx_0"

    num_samples = in_dataset.num_samples
    # heads = [num_samples, num_samples]  # [num_samples]
    # dimensions = [8 * num_smples, 4*num_samples, 2*num_samples]
    heads = [4 * num_samples, 4 * num_samples, 4 * num_samples]
    # heads = [8 * num_samples]
    # dimensions = [8]
    dimensions = [
        4 * num_samples,
        4 * num_samples,
        4 * num_samples,
    ]  # [1]#, num_samples, num_samples]
    #print(heads)
    #print(dimensions)
    # module = GatNodeImputer(in_dim = num_samples + 2,
    #                         heads=heads, gat_dims=dimensions,
    #                         mask_substitute_value=missing_substitute_value, hide_substitute_value=missing_substitute_value,
    #                         nan_substitute_value=missing_substitute_value,
    #                         out_dim=num_samples, use_gatv2=use_gatv2, initial_dense_layers=[8*num_samples, num_samples])
    module = UncertaintyGatNodeImputer(
        in_dim=1+ 2 + len(feature_names) if train_sample_wise else 2 + num_samples + num_samples * len(feature_names),
        heads=heads,
        gat_dims=dimensions,
        mask_substitute_value=missing_substitute_value,
        hide_substitute_value=missing_substitute_value,
        nan_substitute_value=missing_substitute_value,
        out_dim=1 if train_sample_wise else num_samples,
        use_gatv2=use_gatv2,
        initial_dense_layers=[
            8 * num_samples,
            8 * num_samples,
            4 * num_samples,
        ],  # [8 * num_samples, 2 * num_samples]
        dropout=0.2,
        lr=0.001,
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        average_heads = True
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
                TrainingEarlyStopping(
                    monitor=early_stopping_monitor,
                    mode="min",
                    patience=early_stopping_patience,
                )
            ],
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dls)
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
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dls)

    if train_on_partner:
        predict_ds = mask_missing(dataset=in_dataset, molecule_columns={molecule:'abundance', partner_molecule: 'abundance'})
    else:
        predict_ds = mask_missing(dataset=in_dataset, molecule_columns={molecule:'abundance'})

    # pred_dl = DataLoader([(predict_ds, s) for s in in_dataset.sample_names], batch_size=1, collate_fn=collate_fn)
    # predictions = trainer.predict(
    #     module, dataloaders=pred_dl, ckpt_path=None
    # )
    pred_graphs = []
    if train_sample_wise:
        sample_lists = [[s] for s in in_dataset.sample_names]
    else:
        sample_lists = [in_dataset.sample_names]
    for s in sample_lists:
        pred_graphs.append(
            predict_ds.to_dgl_graph(
                feature_columns={
                    mol: ["abundance"] + feature_names for mol in predict_ds.dataset.molecules.keys()
                },
                mappings=[mapping],
                samples=s,
            )
        )
    ntypes = [pred_graph.ntypes for pred_graph in pred_graphs]
    pred_graphs = masked_heterograph_to_homogeneous(
        masked_heterographs=pred_graphs, target="abundance", features=feature_names,
    )
    results = trainer.predict(
        module, DataLoader(pred_graphs, batch_size=1, collate_fn=collator.collate, shuffle=False)
    )
    for predict_graph, res, s, nt in zip(pred_graphs, results, sample_lists, ntypes):
        molecule_index = nt.index(molecule)
        molecule_mask = (predict_graph.ndata[dgl.NTYPE] == molecule_index).type(torch.bool)
        molecule_res = res[molecule_mask]
        masked_molecules = predict_graph.ndata["mask"][molecule_mask].type(torch.bool)
        molecule_ids = predict_graph.ndata[dgl.NID][molecule_mask].type(torch.int64)
        molecule_res = molecule_res[molecule_ids, :]
        masked_molecules = masked_molecules[molecule_ids, :]
        mat_pd = in_dataset.get_samples_value_matrix(molecule=molecule, column="abundance", samples=s)
        mat = mat_pd.to_numpy()
        mat[masked_molecules.numpy()] = molecule_res[masked_molecules].numpy()[:, 0]
        # mat[:, :] = prot_res[:, :].numpy()
        mat_pd.loc[:, :] = mat
        in_dataset.set_samples_value_matrix(
            matrix=mat_pd, molecule=molecule, column="abundance"
        )
        if train_on_partner and partner_result_column is not None:
            partner_index = nt.index(partner_molecule)
            partner_mask = (predict_graph.ndata[dgl.NTYPE] == partner_index).type(torch.bool)
            partner_res = res[partner_mask]
            masked_partner_mols = predict_graph.ndata["mask"][partner_mask].type(torch.bool)
            partner_ids = predict_graph.ndata[dgl.NID][partner_mask].type(torch.int64)
            partner_res = partner_res[partner_ids, :]
            masked_partner_mols = masked_partner_mols[partner_ids, :]
            mat_pd = in_dataset.get_samples_value_matrix(molecule=partner_molecule, column="abundance", samples=s)
            mat = mat_pd.to_numpy()
            mat[masked_partner_mols.numpy()] = partner_res[masked_partner_mols].numpy()[:, 0]
            # mat[:, :] = prot_res[:, :].numpy()
            mat_pd.loc[:, :] = mat
            in_dataset.set_samples_value_matrix(
                matrix=mat_pd, molecule=partner_molecule, column="abundance"
            )
    normalizer.unnormalize(dataset=in_dataset, inplace=True)
    vals = dataset.values[molecule][column]
    res_vals = in_dataset.values[molecule]["abundance"]
    vals.loc[vals.isna(), :] = res_vals.loc[vals.isna(), :]
    if result_column is not None:
        dataset.values[molecule][result_column] = vals
    if partner_result_column is not None:
        vals = dataset.values[partner_molecule][partner_column]
        res_vals = in_dataset.values[partner_molecule]["abundance"]
        vals.loc[vals.isna(), :] = res_vals.loc[vals.isna(), :]
        dataset.values[partner_molecule][partner_result_column] = vals
    if uncertainty_column is not None:
        mat[:, :] = np.nan
        mat[masked_molecules.numpy()] = molecule_res[masked_molecules].numpy()[:, 1]
        mat_pd.loc[:, :] = mat
        dataset.set_samples_value_matrix(
            matrix=mat_pd, molecule=molecule, column=uncertainty_column
        )
    return vals
