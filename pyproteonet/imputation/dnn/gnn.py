from typing import Any, Optional, List
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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ...data.dataset import Dataset
from ...masking.masked_dataset_generator import MaskedDatasetGenerator
from ...masking.random import mask_molecule_values_random_non_missing
from ...lightning.console_logger import ConsoleLogger
from ...normalization.dnn_normalizer import DnnNormalizer
from ...masking.missing_values import mask_missing
from ...data.masked_dataset import MaskedDataset


class ImputationModuleBackup(L.LightningModule):
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
        mask_value=-10,
    ):
        super().__init__()
        self.molecule = molecule
        self.partner_molecule = partner_molecule
        self.mapping = mapping
        dense_layers = []
        last_dim = in_dim
        for dim in layers:
            dense_layers.append(nn.Linear(last_dim, dim))
            dense_layers.append(nn.Dropout(p=dropout))
            dense_layers.append(nn.LeakyReLU())
            last_dim = dim
        dense_layers.append(nn.Linear(last_dim, 2 * in_dim))
        self.partner_fc_model = nn.Sequential(*dense_layers)
        partner_fc_out_dim = 2 * in_dim
        self.molecule_gat = GATv2Conv(
            in_feats=(partner_fc_out_dim, in_dim),
            out_feats=2 * in_dim,
            num_heads=gat_heads,
            feat_drop=dropout,
            attn_drop=dropout,
        )
        self.molecule_linear = nn.Linear(2 * in_dim * gat_heads, 2 * in_dim)
        self.loss_fn = torch.nn.GaussianNLLLoss()
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
        molecule_inputs = abundance[self.molecule]
        output = self.partner_fc_model(partner_inputs)
        output = self.molecule_gat(graph, (output, molecule_inputs))
        output = output.view(output.shape[0], -1)
        output = nn.functional.leaky_relu(output)
        output = self.molecule_linear(output)
        output_shape = list(output.shape)
        output_shape[-1] = int(output_shape[-1] / 2)
        output = output.reshape(*output_shape, 2)
        return output

    def compute_loss(self, graph) -> torch.tensor:
        abundance = graph.ndata["abundance"]
        masks = graph.ndata["mask"]
        molecule_mask = masks[self.molecule]
        molecule_gt = abundance[self.molecule].detach().clone()
        assert torch.isnan(molecule_gt[molecule_mask]).sum() == 0
        molecule_input = abundance[self.molecule]
        molecule_input[molecule_mask] = self.mask_value
        output = self(graph)
        pred = output[molecule_mask][:, 0]
        var = torch.exp(output[molecule_mask][:, 1])
        target = molecule_gt[molecule_mask]
        # loss = torch.nn.functional.mse_loss(output, inputs.mean(dim=-1, keepdim=True))
        loss = self.loss_fn(pred, target=target, var=var)
        return loss

    def training_step(self, graph, batch_idx):
        loss = self.compute_loss(graph)
        # loss = torch.nn.functional.mse_loss(output[mask], gt[mask])
        self.log("train_loss", loss.item())
        # self.log('target_mean', target.mean().item())
        # self.log('target_std', target.std().item())
        # self.log('pred_mean', pred.mean().item())
        # self.log('pred_std', pred.std().item())
        # self.log('var_mean', var.mean().item())
        # self.log('var_std', var.std().item())
        # inputs = inputs.detach().cpu()
        # inputs[inputs==-3] = np.nan
        # self.log('mean_diff', torch.nanmean((torch.nanmean(inputs.detach().cpu(), axis=1) - output.detach().cpu())**2))
        # output = output.detach().cpu().flatten()
        # self.log('noisy_std', output[:200].mean())
        # self.log('non_noisy_std', output[200:].mean())
        return loss

    def validation_step(self, graph, batch_idx):
        loss = self.compute_loss(graph)
        self.log("val_loss", loss.item(), batch_size=1)

    def predict_step(self, graph, batch_idx, dataloader_idx=0):
        res = self(graph)
        return res[:, :, 0], torch.exp(res[:, :, 1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ImputationModule(L.LightningModule):
    def __init__(
        self,
        molecule,
        partner_molecule,
        mapping,
        in_dim,
        layers,
        molecule_loss_coefficent=1,
        partner_loss_coefficent=1,
        lr=0.1,
        dropout=0.2,
        gat_heads=10,
        gat_dim=64,
        mask_value=-10,
    ):
        super().__init__()
        self.molecule = molecule
        self.partner_molecule = partner_molecule
        self.mapping = mapping
        self.molecule_loss_coefficent = molecule_loss_coefficent
        self.partner_loss_coefficent = partner_loss_coefficent
        self.etype = (partner_molecule, mapping, molecule)
        self.etype_inverse = (molecule, mapping, partner_molecule)
        dense_layers = []
        last_dim = in_dim
        for dim in layers:
            dense_layers.append(nn.Linear(last_dim, dim))
            dense_layers.append(nn.Dropout(p=dropout))
            dense_layers.append(nn.LeakyReLU())
            last_dim = dim
        dense_layers.append(nn.Linear(last_dim, 2 * in_dim))
        self.partner_fc_model = nn.Sequential(*dense_layers)
        partner_fc_out_dim = 2 * in_dim
        self.molecule_gat = HeteroGraphConv(
            {
                self.etype: GATv2Conv(
                    in_feats=(partner_fc_out_dim, in_dim),
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
                    in_feats=(gat_dim, partner_fc_out_dim),
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
        self.partner_linear = nn.Linear(gat_dim, 2 * in_dim)
        self.loss_fn = torch.nn.GaussianNLLLoss()
        self.lr = lr
        self.mask_value = mask_value

    def forward(self, graph):
        # inverse_graph = graph.reverse()
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
        molecule_inputs = abundance[self.molecule]
        partner_vec = self.partner_fc_model(partner_inputs)
        mol_vec = nn.functional.leaky_relu(
            self.molecule_gat(graph, ({self.partner_molecule:partner_vec}, {self.molecule:molecule_inputs}))[self.molecule].mean(dim=-2)
        )
        partner_vec = nn.functional.leaky_relu(
            self.partner_gat(graph, ({self.molecule:mol_vec}, {self.partner_molecule:partner_vec}))[self.partner_molecule].mean(dim=-2)
        )
        mol_vec = nn.functional.leaky_relu(
            self.molecule_gat2(graph, ({self.partner_molecule:partner_vec}, {self.molecule:mol_vec}))[self.molecule].mean(dim=-2)
        )
        # output = output.view(output.shape[0], -1)
        # reshape molecule vector
        mol_vec = self.molecule_linear(mol_vec)
        mol_shape = list(mol_vec.shape)
        mol_shape[-1] = int(mol_shape[-1] / 2)
        mol_vec = mol_vec.reshape(*mol_shape, 2)
        # reshape partner vector
        partner_vec = self.partner_linear(partner_vec)
        partner_shape = list(partner_vec.shape)
        partner_shape[-1] = int(partner_shape[-1] / 2)
        partner_vec = partner_vec.reshape(*partner_shape, 2)
        return mol_vec, partner_vec

    def compute_loss(self, graph) -> torch.tensor:
        abundance = graph.ndata["abundance"]
        masks = graph.ndata["mask"]
        molecule_mask = masks[self.molecule]
        molecule_gt = abundance[self.molecule].detach().clone()
        assert torch.isnan(molecule_gt[molecule_mask]).sum() == 0
        molecule_input = abundance[self.molecule]
        molecule_input[molecule_mask] = self.mask_value
        mol_target = molecule_gt[molecule_mask]

        partner_mask = masks[self.partner_molecule]
        partner_gt = abundance[self.partner_molecule].detach().clone()
        assert torch.isnan(partner_gt[partner_mask]).sum() == 0
        partner_input = abundance[self.partner_molecule]
        partner_input[partner_mask] = self.mask_value
        partner_target = partner_gt[partner_mask]

        mol_vec, partner_vec = self(graph)
        mol_pred = mol_vec[molecule_mask][:, 0]
        partner_pred = partner_vec[partner_mask][:, 0]
        partner_var = torch.exp(partner_vec[partner_mask][:, 1])
        mol_var = torch.exp(mol_vec[molecule_mask][:, 1])
        # loss = torch.nn.functional.mse_loss(output, inputs.mean(dim=-1, keepdim=True))
        loss = self.molecule_loss_coefficent * self.loss_fn(
            mol_pred, target=mol_target, var=mol_var
        ) + self.partner_loss_coefficent * self.loss_fn(
            partner_pred, target=partner_target, var=partner_var
        )
        return loss

    def training_step(self, graph, batch_idx):
        loss = self.compute_loss(graph)
        # loss = torch.nn.functional.mse_loss(output[mask], gt[mask])
        self.log("train_loss", loss.item())
        # self.log('target_mean', target.mean().item())
        # self.log('target_std', target.std().item())
        # self.log('pred_mean', pred.mean().item())
        # self.log('pred_std', pred.std().item())
        # self.log('var_mean', var.mean().item())
        # self.log('var_std', var.std().item())
        # inputs = inputs.detach().cpu()
        # inputs[inputs==-3] = np.nan
        # self.log('mean_diff', torch.nanmean((torch.nanmean(inputs.detach().cpu(), axis=1) - output.detach().cpu())**2))
        # output = output.detach().cpu().flatten()
        # self.log('noisy_std', output[:200].mean())
        # self.log('non_noisy_std', output[200:].mean())
        return loss

    def validation_step(self, graph, batch_idx):
        loss = self.compute_loss(graph)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def impute_gnn(
    dataset: Dataset,
    molecule: str,
    column: str,
    mapping: str,
    partner_column: str,
    molecule_result_column: Optional[str] = None,
    molecule_uncertainty_column: Optional[str] = None,
    partner_result_column: Optional[str] = None,
    partner_uncertainty_column: Optional[str] = None,
    molecule_coefficient=1,
    partner_coefficient=1,
    max_epochs: int = 5000,
) -> pd.Series:
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )

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

    ds = dataset.copy(columns={molecule: column, partner_molecule: partner_column})
    ds.rename_columns(
        columns={
            molecule: {column: "abundance"},
            partner_molecule: {partner_column: "abundance"},
        },
        inplace=True,
    )
    normalizer = DnnNormalizer(columns=["abundance"])
    normalizer.normalize(dataset=ds, inplace=True)

    validation_ids = ds.values[molecule]["abundance"]
    validation_ids = validation_ids[~validation_ids.isna()].sample(frac=0.1).index
    partner_validation_ids = ds.values[partner_molecule]["abundance"]
    partner_validation_ids = (
        partner_validation_ids[~partner_validation_ids.isna()].sample(frac=0.1).index
    )
    validation_set = MaskedDataset.from_ids(
        dataset=ds,
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

    mask_ds = MaskedDatasetGenerator(datasets=[ds], generator_fn=masking_fn)

    collator = GraphCollator()

    def collate(mds: List[MaskedDataset]):
        res = []
        for md in mds:
            graph = md.to_dgl_graph(
                molecule_features={
                    molecule: "abundance",
                    partner_molecule: "abundance",
                },
                mappings=[mapping],
                mapping_directions={mapping: (partner_molecule, molecule)},
                make_bidirectional=True,
                cache=True,
            )
            res.append(graph)
        return collator.collate(res)

    train_dl = DataLoader(mask_ds, batch_size=1, collate_fn=collate)
    validation_dl = DataLoader([validation_set], batch_size=1, collate_fn=collate)

    in_dim = dataset.num_samples
    model = ImputationModule(
        molecule=molecule,
        partner_molecule=partner_molecule,
        mapping=mapping,
        in_dim=in_dim,
        layers=[8 * in_dim, 8 * in_dim, 4 * in_dim],
        gat_heads=2 * in_dim,
        gat_dim=4 * in_dim,
        molecule_loss_coefficent=molecule_coefficient,
        partner_loss_coefficent=partner_coefficient,
        dropout=0.1,
        lr=0.001,
    )

    trainer = L.Trainer(
        logger=ConsoleLogger(),
        log_every_n_steps=10,
        check_val_every_n_epoch=10,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
    )
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=validation_dl)

    missing_molecule = ds.values[molecule]["abundance"]
    missing_molecule = missing_molecule[missing_molecule.isna()].index
    missing_partner = ds.values[partner_molecule]["abundance"]
    missing_partner = missing_partner[missing_partner.isna()].index
    missing_mds = MaskedDataset.from_ids(
        dataset=ds,
        mask_ids={molecule: missing_molecule, partner_molecule: missing_partner},
    )
    pred_dl = DataLoader([missing_mds], batch_size=1, collate_fn=collate)
    pred_mol, uncertainty_mol, pred_partner, uncertainty_partner = trainer.predict(
        model=model, dataloaders=pred_dl, ckpt_path=None
    )[0]
    pred_mol, uncertainty_mol = pred_mol.numpy(), uncertainty_mol.numpy()
    pred_partner, uncertainty_partner = (
        pred_partner.numpy(),
        uncertainty_partner.numpy(),
    )

    missing_mds.set_samples_value_matrix(
        matrix=pred_mol, molecule=molecule, column="abundance", only_set_masked=True
    )
    missing_mds.set_samples_value_matrix(
        matrix=pred_partner,
        molecule=partner_molecule,
        column="abundance",
        only_set_masked=True,
    )
    normalizer.unnormalize(dataset=ds, inplace=True)

    missing_mds.set_samples_value_matrix(
        matrix=uncertainty_mol * normalizer.stds[molecule]["abundance"],
        molecule=molecule,
        column="uncertainty",
        only_set_masked=True,
    )
    missing_mds.set_samples_value_matrix(
        matrix=uncertainty_partner * normalizer.stds[partner_molecule]["abundance"],
        molecule=partner_molecule,
        column="uncertainty",
        only_set_masked=True,
    )

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
