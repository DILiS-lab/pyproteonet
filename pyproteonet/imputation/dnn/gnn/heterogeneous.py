from typing import Any, Optional, List, Tuple
import random
from lightning.pytorch.utilities.types import STEP_OUTPUT

import numpy as np
import pandas as pd
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dgl.nn.pytorch.conv import GATConv, GATv2Conv
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch.utils import Sequential as DglSequential
from dgl.dataloading import GraphCollator
import dgl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import Logger

from ....data.dataset import Dataset
from ....masking.masked_dataset_generator import MaskedDatasetGenerator
from ....lightning.console_logger import ConsoleLogger
from ....lightning.training_early_stopping import TrainingEarlyStopping
from ....processing.standardizer import Standardizer
from ....masking.missing_values import mask_missing
from ....masking.masked_dataset import MaskedDataset

class ImputationModule(L.LightningModule):
    def __init__(
        self,
        molecule,
        partner_molecule,
        mapping,
        in_dim,
        layers,
        lr=0.1,
        dropout=0.2,
        gat_heads=10,
        gat_dim=64,
        mask_value=-2,
        num_embeddings: Optional[int] = None,
        embedding_dim: Optional[int] = 64,
    ):
        super().__init__()
        self.molecule = molecule
        self.partner_molecule = partner_molecule
        self.mapping = mapping
        self.etype = (partner_molecule, mapping, molecule)
        self.etype_inverse = (molecule, mapping, partner_molecule)
        dense_layers_partner = []
        dense_layers_molecule = []
        fc_out_dim = 2 * in_dim
        if num_embeddings is not None and embedding_dim is not None:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        else:
            self.embedding = None
            embedding_dim = 0
        for dense_layers, fc_in_dim in [(dense_layers_partner, in_dim), (dense_layers_molecule, embedding_dim)]:
            last_dim = fc_in_dim
            for dim in layers:
                dense_layers.append(nn.Linear(last_dim, dim))
                dense_layers.append(nn.Dropout(p=dropout))
                dense_layers.append(nn.LeakyReLU())
                last_dim = dim
            dense_layers.append(nn.Linear(last_dim, fc_out_dim))
        self.molecule_fc_model = nn.Sequential(*dense_layers_molecule)
        self.partner_fc_model = nn.Sequential(*dense_layers_partner)
        self.molecule_gat = HeteroGraphConv(
            {
                self.etype: GATv2Conv(
                    in_feats=(fc_out_dim, fc_out_dim),
                    out_feats=gat_dim,
                    num_heads=gat_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            }
        )
        self.partner_gat = HeteroGraphConv(
            {
                self.etype_inverse: GATv2Conv(
                    in_feats=(gat_dim, fc_out_dim),
                    out_feats=gat_dim,
                    num_heads=gat_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=True,
                )
            }
        )
        self.molecule_gat2 = HeteroGraphConv(
            {
                self.etype: GATv2Conv(
                    in_feats=(gat_dim, gat_dim),
                    out_feats=gat_dim,
                    num_heads=gat_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                )
            }
        )
        self.molecule_linear = nn.Linear(gat_dim, 2 * in_dim)
        self.partner_linear = nn.Linear(gat_dim + fc_out_dim, 2 * in_dim)
        self.loss_fn = torch.nn.GaussianNLLLoss(eps=1e-4)
        self.lr = lr
        self.mask_value = mask_value

    def forward(self, graph):
        abundance = graph.ndata["abundance"]
        hidden = graph.ndata["hidden"]
        for key, hide in hidden.items():
            abundance[key][hide] = self.mask_value
        masks = graph.ndata["mask"]
        for key, mask in masks.items():
            abundance[key][mask] = self.mask_value
        for key, ab in abundance.items():
            ab[torch.isnan(ab)] = self.mask_value
        partner_inputs = abundance[self.partner_molecule]
        partner_fc_vec = self.partner_fc_model(partner_inputs)
        molecule_fc_vec = self.molecule_fc_model(self.embedding(graph.nodes(self.molecule).int()))
        mol_vec = nn.functional.leaky_relu(
            self.molecule_gat(graph, ({self.partner_molecule:partner_fc_vec}, {self.molecule:molecule_fc_vec}))[self.molecule].mean(dim=-2)
        )
        partner_vec = nn.functional.leaky_relu(
            self.partner_gat(graph, ({self.molecule:mol_vec}, {self.partner_molecule:partner_fc_vec}))[self.partner_molecule].mean(dim=-2)
        )
        mol_vec = nn.functional.leaky_relu(
            self.molecule_gat2(graph, ({self.partner_molecule:partner_vec}, {self.molecule:mol_vec}))[self.molecule].mean(dim=-2)
        )
        mol_vec = self.molecule_linear(mol_vec)
        mol_shape = list(mol_vec.shape)
        mol_shape[-1] = int(mol_shape[-1] / 2)
        mol_vec = mol_vec.reshape(*mol_shape, 2)
        # reshape partner vector
        partner_vec = torch.cat((partner_vec, partner_fc_vec), dim=-1)
        partner_vec = self.partner_linear(partner_vec)
        partner_shape = list(partner_vec.shape)
        partner_shape[-1] = int(partner_shape[-1] / 2)
        partner_vec = partner_vec.reshape(*partner_shape, 2)
        if mol_vec.isnan().any().item() or partner_vec.isnan().any().item():
            print('nan prediction')
        return mol_vec, partner_vec

    def compute_loss(self, graph, partner_loss: bool = True) -> torch.tensor:
        abundance = graph.ndata["abundance"]
        masks = graph.ndata["mask"]
        molecule_mask = masks[self.molecule]
        num_masked_molecule = molecule_mask.sum()
        molecule_gt = abundance[self.molecule].detach().clone()
        assert torch.isnan(molecule_gt[molecule_mask]).sum() == 0
        molecule_input = abundance[self.molecule]
        molecule_input[molecule_mask] = self.mask_value
        mol_target = molecule_gt[molecule_mask]

        partner_mask = masks[self.partner_molecule]
        num_masked_partner = partner_mask.sum()
        partner_gt = abundance[self.partner_molecule].detach().clone()
        assert torch.isnan(partner_gt[partner_mask]).sum() == 0
        partner_gt[torch.isnan(partner_gt)] = self.mask_value
        partner_input = abundance[self.partner_molecule]
        partner_input[partner_mask] = self.mask_value
        partner_target = partner_gt[partner_mask]

        mol_vec, partner_vec = self(graph)
        mol_pred = mol_vec[molecule_mask][:, 0]
        partner_pred = partner_vec[partner_mask][:, 0]
        partner_var = torch.exp(partner_vec[partner_mask][:, 1])
        mol_var = torch.exp(mol_vec[molecule_mask][:, 1])
        self.log("num_masked_molecule", num_masked_molecule.item(), on_step=False, on_epoch=True, batch_size=1)
        self.log("num_masked_partner", num_masked_partner.item(), on_step=False, on_epoch=True, batch_size=1)
        molecule_loss_coefficient = 1
        partner_loss_coefficient = 1
        loss = molecule_loss_coefficient * self.loss_fn(
            mol_pred, target=mol_target, var=mol_var
        )
        self.log("molecule_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1)
        if partner_loss and partner_loss_coefficient > 0:
            partner_loss = partner_loss_coefficient * self.loss_fn(
                partner_pred, target=partner_target, var=partner_var
            )
            loss += partner_loss
            self.log("partner_loss", partner_loss.item(), on_step=False, on_epoch=True, batch_size=1)
        loss = torch.min(loss, torch.max(partner_target.max(), mol_target.max()))
        return loss

    def training_step(self, graph, batch_idx):
        loss = self.compute_loss(graph)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, graph, batch_idx):
        loss = self.compute_loss(graph, partner_loss=False)
        self.log("val_loss", loss.item(), batch_size=1)

    def predict_step(self, graph, batch_idx, dataloader_idx=0):
        mol_vec, partner_vec = self(graph)
        return (
            mol_vec[:, :, 0],
            torch.exp(mol_vec[:, :, 1]),
            partner_vec[:, :, 0],
            torch.exp(partner_vec[:, :, 1]),
        )

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def impute_heterogeneous_gnn(
    dataset: Dataset,
    molecule: str,
    column: str,
    mapping: str,
    partner_column: str,
    molecule_result_column: Optional[str] = None,
    molecule_uncertainty_column: Optional[str] = None,
    partner_result_column: Optional[str] = None,
    partner_uncertainty_column: Optional[str] = None,
    max_epochs: int = 5000,
    training_fraction: float = 0.25,
    train_sample_wise: bool = False,
    log_every_n_steps: Optional[int] = None,
    early_stopping_patience: int = 7,
    logger: Optional[Logger] = None,
    epoch_size: int = 30,
    missing_substitute_value: int = -2
) -> pd.Series:
    """Impute missing values using a heterogeneous graph neural network applied on the molecule graph created from two molecule types like proteins and their assigned peptides.

    Args:
        dataset (Dataset): The dataset to impute.
        molecule (str): The main molecule type to impute (e.g. "protein").
        column (str): The value column of the main molecule type to impute (e.g. "abundance").
        mapping (str): The name of the mapping, connecting the main molecule type with a partner molecule type (e.g. "protein-peptide").
        partner_column (str): The value column of the partner molecule type to impute.
        molecule_result_column (Optional[str], optional): If given imputed values for the molecule are stored under this name. Defaults to None.
        molecule_uncertainty_column (Optional[str], optional): If given predicted uncertainty values for the main molecule are stored under this name. Defaults to None.
        partner_result_column (Optional[str], optional): If given imputed values for the partner molecule are stored under this name. Defaults to None.
        partner_uncertainty_column (Optional[str], optional): If given predicted uncertainty values for the main molecule are stored under this name. Defaults to None.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 5000.
        training_fraction (float, optional): Mean fraction of molecules masked during training (The masking fraction for every epoch is randomly drawn from the (0.5 * training_fraction, 1.5 * training_fraction) interval). Defaults to 0.25.
        train_sample_wise (bool, optional): Whether a training step operates only on a single sample or the whole dataset. Defaults to False.
        log_every_n_steps (Optional[int], optional): How often to log during training. Defaults to None.
        early_stopping_patience (int, optional): Number of epochs after which the training is stopped if the training loss does not improve. Defaults to 7.
        logger (Optional[Logger], optional): The lightning logger used for logging. If not given logs will be printed to consose. Defaults to None.
        epoch_size (int, optional): Number of training runs on the dataset that make up an epoch. Defaults to 30.
        missing_substitute_value (float, optional): Value to replace missing or masked values with. Defaults to -3.
    Returns:
        pd.Series: the imputed values.
    """
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    if log_every_n_steps is None:
        if train_sample_wise:
            log_every_n_steps = dataset.num_samples
        else:
            log_every_n_steps = 1

    ds = dataset.copy(columns={molecule: [column],
                               partner_molecule: [partner_column]})
    ds.rename_columns(
        columns={
            molecule: {column: "abundance"},
            partner_molecule: {partner_column: "abundance"},
        },
        inplace=True,
    )
    normalizer = Standardizer(columns=["abundance"])
    normalizer.standardize(dataset=ds, inplace=True)

    mapping_df = ds.mappings[mapping].df
    partner_mask_ids = ds.values[partner_molecule]["abundance"]
    partner_mask_ids = partner_mask_ids[partner_mask_ids.index.get_level_values('id').isin(mapping_df.index.get_level_values(partner_molecule).unique())]
    partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
    def masking_fn(in_ds):
        rng = np.random.default_rng()
        epoch_masking_fraction = rng.uniform(0.5 * training_fraction, 1.5 * training_fraction)
        molecule_mask_ids = in_ds.values[molecule]["abundance"]
        molecule_mask_ids = molecule_mask_ids[~molecule_mask_ids.isna()].index
        partner_ids = (
            partner_mask_ids
            .sample(frac=epoch_masking_fraction)
            .index
        )
        return MaskedDataset.from_ids(
            dataset=in_ds,
            mask_ids={molecule: molecule_mask_ids, partner_molecule: partner_ids},
        )

    mask_ds = MaskedDatasetGenerator(datasets=[ds], generator_fn=masking_fn, sample_wise=train_sample_wise, epoch_size_multiplier=epoch_size)

    collator = GraphCollator()

    def collate(mds: List[Tuple[MaskedDataset, List[str]]]):
        res = []
        for md, samples in mds:
            graph = md.to_dgl_graph(
                feature_columns={
                    molecule: "abundance",
                    partner_molecule: "abundance",
                },
                mappings=[mapping],
                mapping_directions={mapping: (partner_molecule, molecule)},
                make_bidirectional=True,
                samples=samples,
            )
            res.append(graph)
        return collator.collate(res)

    train_dl = DataLoader(mask_ds, batch_size=1, collate_fn=collate)
    graph = list(train_dl)[0]
    num_embeddings = int(graph.num_nodes(ntype=molecule))

    num_samples = dataset.num_samples
    in_dim = 1 if train_sample_wise else num_samples
    model = ImputationModule(
        molecule=molecule,
        partner_molecule=partner_molecule,
        mapping=mapping,
        in_dim=in_dim,
        layers=[8 * num_samples, 8 * num_samples, 4 * num_samples],
        gat_heads=2 * num_samples,
        gat_dim=4 * num_samples,
        dropout=0.1,
        lr=0.001,
        num_embeddings=num_embeddings,
        embedding_dim=max(4, ds.num_samples // 2),
        mask_value=missing_substitute_value
    )
    if logger is None:
        logger = ConsoleLogger()
    trainer = L.Trainer(
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        callbacks=[TrainingEarlyStopping(monitor="train_loss", mode="min", patience=early_stopping_patience)],
        gradient_clip_val=1,
    )
    trainer.fit(model=model, train_dataloaders=train_dl)

    missing_molecule = ds.values[molecule]["abundance"]
    missing_molecule = missing_molecule[missing_molecule.isna()].index
    missing_partner = ds.values[partner_molecule]["abundance"]
    missing_partner = missing_partner[missing_partner.isna()].index
    missing_mds = MaskedDataset.from_ids(
        dataset=ds,
        mask_ids={molecule: missing_molecule, partner_molecule: missing_partner},
    )
    if train_sample_wise:
        sample_names = [[s] for s in ds.sample_names]
        pred_dl = DataLoader([(missing_mds, s) for s in sample_names], batch_size=1, collate_fn=collate)
    else:
        sample_names = [None]
        pred_dl = DataLoader([(missing_mds, None)], batch_size=1, collate_fn=collate)
    for (pred_mol, uncertainty_mol, pred_partner, uncertainty_partner), s in zip(trainer.predict(
        model=model, dataloaders=pred_dl, ckpt_path=None
    ), sample_names):
        pred_mol, uncertainty_mol = pred_mol.numpy(), uncertainty_mol.numpy()
        pred_partner, uncertainty_partner = (
            pred_partner.numpy(),
            uncertainty_partner.numpy(),
        )

        missing_mds.set_samples_value_matrix(
            matrix=pred_mol, molecule=molecule, column="abundance", only_set_masked=True, samples=s
        )
        missing_mds.set_samples_value_matrix(
            matrix=pred_partner,
            molecule=partner_molecule,
            column="abundance",
            only_set_masked=True, samples=s
        )

        missing_mds.set_samples_value_matrix(
            matrix=uncertainty_mol * normalizer.stds[molecule]["abundance"],
            molecule=molecule,
            column="uncertainty",
            only_set_masked=True, samples=s
        )
        missing_mds.set_samples_value_matrix(
            matrix=uncertainty_partner * normalizer.stds[partner_molecule]["abundance"],
            molecule=partner_molecule,
            column="uncertainty",
            only_set_masked=True, samples=s
        )

    normalizer.unstandardize(dataset=ds, inplace=True)

    res_molecule = dataset.values[molecule][column]
    res_partner = dataset.values[partner_molecule][column]
    mask_molecule = res_molecule[res_molecule.isna()].index
    mask_partner = res_partner[res_partner.isna()].index
    res_molecule.loc[mask_molecule] = ds.values[molecule]["abundance"].loc[
        mask_molecule
    ]
    res_partner.loc[mask_partner] = ds.values[partner_molecule]["abundance"].loc[
        mask_partner
    ]
    if molecule_result_column is not None:
        dataset.values[molecule][molecule_result_column] = res_molecule
    if partner_result_column is not None:
        dataset.values[partner_molecule][partner_result_column] = res_partner
    if molecule_uncertainty_column is not None:
        uncertainty_mol = ds.values[molecule]["uncertainty"]
        uncertainty_mol[~uncertainty_mol.index.isin(mask_molecule)] = np.nan
        dataset.values[molecule][molecule_uncertainty_column] = uncertainty_mol
    if partner_uncertainty_column is not None:
        uncertainty_partner = ds.values[partner_molecule]["uncertainty"]
        uncertainty_partner[~uncertainty_partner.index.isin(mask_partner)] = np.nan
        dataset.values[partner_molecule][
            partner_uncertainty_column
        ] = uncertainty_partner
    return res_molecule
