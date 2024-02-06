from typing import Any, Optional, List, Tuple
import random
import functools

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
        embedding_molecule_key: str,
        embedding_partner_key: str,
        num_samples: int,
        lr=0.1,
        dropout=0.2,
        gat_heads=10,
        gat_dim=64,
        mask_value=-2,
        embedding_in_dim: int = 1024,
        embedding_shrink_dim: Optional[int] = 32,
        molecule_embedding_shrink_dim: int = 4,
        normalizer: Optional[Standardizer] = None,
        use_sample_embeddings: bool = True,
    ):
        super().__init__()
        self.molecule = molecule
        self.partner_molecule = partner_molecule
        self.mapping = mapping
        self.normalizer = normalizer
        self.use_sample_embeddings = use_sample_embeddings
        self.etype = (partner_molecule, mapping, molecule)
        self.etype_inverse = (molecule, mapping, partner_molecule)
        fc_out_dim = 2 * gat_dim
        self.embedding_molecule_key = embedding_molecule_key
        self.embedding_partner_key = embedding_partner_key
        self.embedding_shrink_dim = embedding_shrink_dim
        self.molecule_embedding_map = nn.Sequential(nn.Linear(embedding_in_dim, embedding_shrink_dim), nn.Dropout(p=dropout))
        self.partner_embedding_map = nn.Sequential(nn.Linear(embedding_in_dim, embedding_shrink_dim))
        if self.use_sample_embeddings:
            sample_emb_dim = 128
            self.samples_embeddings = nn.Embedding(num_samples, sample_emb_dim)
        else:
            sample_emb_dim = 0
        dense_layers_partner = []
        last_dim = in_dim
        for dim in [in_dim + embedding_shrink_dim + sample_emb_dim, fc_out_dim, fc_out_dim]:
            dense_layers_partner.append(nn.Linear(last_dim, dim))
            # dense_layers.append(nn.Dropout(p=dropout))
            dense_layers_partner.append(nn.LeakyReLU())
            last_dim = dim
        # dense_layers_molecule.append(nn.Linear(embedding_shrink_dim + sample_emb_dim, fc_out_dim))
        dense_layers_partner.append(nn.Linear(last_dim, fc_out_dim))
        #dense_layers.append(nn.Dropout(p=dropout))
        self.partner_abundance_fc = nn.Sequential(*dense_layers_partner)
        partner_latent_dim = fc_out_dim
        self.molecule_gat = HeteroGraphConv(
            {
                self.etype: GATv2Conv(
                    in_feats=(
                        partner_latent_dim,
                        embedding_shrink_dim,
                    ),
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
                    in_feats=(gat_dim + embedding_shrink_dim, partner_latent_dim),
                    out_feats=gat_dim,
                    num_heads=gat_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    allow_zero_in_degree=False,
                )
            }
        )
        self.partner_linear_intermediat = []
        last_dim = gat_dim + partner_latent_dim
        #last_dim = partner_latent_dim
        for dim in [gat_dim, gat_dim]:
            self.partner_linear_intermediat.append(nn.Linear(last_dim, dim))
            self.partner_linear_intermediat.append(nn.LeakyReLU())
            last_dim = dim
        self.partner_linear_intermediat.append(nn.Linear(last_dim, gat_dim))
        self.partner_linear_intermediat = nn.Sequential(*self.partner_linear_intermediat)

        # self.molecule_gat2 = HeteroGraphConv(
        #     {
        #         self.etype: GATv2Conv(
        #             in_feats=(gat_dim + partner_latent_dim, embedding_in_dim),
        #             out_feats=gat_dim,
        #             num_heads=gat_heads,
        #             feat_drop=dropout,
        #             attn_drop=dropout,
        #         )
        #     }
        # )
        #self.molecule_linear = nn.Linear(gat_dim + embedding_shrink_dim, 2 * in_dim)
        self.partner_linear = nn.Linear(gat_dim + partner_latent_dim, 2 * in_dim)
        self.molecule_linear = nn.Linear(embedding_shrink_dim, 2 * in_dim)
        # self.partner_linear = nn.Linear(partner_latent_dim, 2 * in_dim)

        self.loss_fn = torch.nn.GaussianNLLLoss(eps=1e-4)
        self.lr = lr
        self.mask_value = mask_value

    def forward(self, batch, abundance_column: str = "abundance"):
        # inverse_graph = graph.reverse()
        graph, sample_ids = batch
        if self.use_sample_embeddings:
            sample_ids = sample_ids[0]
            sample_emb = self.samples_embeddings(
                torch.tensor(sample_ids, dtype=torch.long, device=graph.device)
            ).mean(axis=0)
        abundance = graph.ndata[abundance_column]
        hidden = graph.ndata["hidden"]
        for key, hide in hidden.items():
            abundance[key][hide] = self.mask_value
        masks = graph.ndata["mask"]
        for key, mask in masks.items():
            abundance[key][mask] = self.mask_value
        for key, ab in abundance.items():
            ab[torch.isnan(ab)] = self.mask_value
        # print(
        #     "inputs",
        #     torch.mean(abundance[self.molecule][:1000]).item(),
        #     torch.mean(abundance[self.partner_molecule][:1000]).item(),
        # )
        partner_inputs = abundance[self.partner_molecule]
        partner_latent = [
            self.partner_embedding_map(
                graph.ndata[self.embedding_partner_key][self.partner_molecule]
            ),
            partner_inputs,
        ]
        if self.use_sample_embeddings:
            sample_emb_partner = sample_emb.repeat(partner_latent[0].shape[0], 1)
            partner_latent.append(sample_emb_partner)
        partner_latent = torch.cat(partner_latent, dim=-1)
        partner_latent = self.partner_abundance_fc(partner_inputs)

        # molecule_inputs = abundance[self.molecule]
        mol_emb = self.molecule_embedding_map(
            graph.ndata[self.embedding_molecule_key][self.molecule]
        )
        molecule_inputs = [mol_emb]
        #if self.use_sample_embeddings:
        #    sample_emb_mol = sample_emb.repeat(molecule_inputs[0].shape[0], 1)
        #    molecule_inputs.append(sample_emb_mol)
        molecule_inputs = torch.cat(molecule_inputs, dim=-1)
        molecule_latent = molecule_inputs  # self.molecule_fc_model(molecule_inputs)

        # partner_vec = partner_latent
        # mol_vec = molecule_latent

        mol_agg = nn.functional.leaky_relu(
            self.molecule_gat(
                graph,
                (
                    {self.partner_molecule: partner_latent},
                    {self.molecule: molecule_latent},
                ),
            )[self.molecule].mean(dim=-2)
        )

        mol_vec = torch.cat((mol_agg, molecule_latent), dim=-1)
        partner_vec = nn.functional.leaky_relu(
            self.partner_gat(
                graph,
                ({self.molecule: mol_vec}, {self.partner_molecule: partner_latent}),
            )[self.partner_molecule].mean(dim=-2)
        )
        partner_vec = torch.cat((partner_vec, partner_latent), dim=-1)

        #partner_vec = self.partner_linear_intermediat(partner_latent)

        # mol_vec = nn.functional.leaky_relu(
        #     self.molecule_gat2(
        #         graph, ({self.partner_molecule: partner_vec}, {self.molecule: molecule_latent})
        #     )[self.molecule].mean(dim=-2)
        # )

        partner_vec = self.partner_linear(partner_vec)
        partner_shape = list(partner_vec.shape)
        partner_shape[-1] = int(partner_shape[-1] / 2)
        partner_vec = partner_vec.reshape(*partner_shape, 2)
        # mol_vec = torch.cat((mol_vec, molecule_fc_vec), dim=-1)
        # output = output.view(output.shape[0], -1)
        # reshape molecule vector
        # mol_vec = torch.cat((mol_vec, molecule_fc_vec), dim=-1)
        mol_vec = self.molecule_linear(molecule_latent)
        mol_shape = list(mol_vec.shape)
        mol_shape[-1] = int(mol_shape[-1] / 2)
        mol_vec = mol_vec.reshape(*mol_shape, 2)
        # reshape partner vector
        if mol_vec.isnan().any().item() or partner_vec.isnan().any().item():
            print("nan prediction")
        # print('outputs', mol_vec.mean().item(), partner_vec.mean().item())
        return mol_vec, partner_vec

    def compute_loss(
        self,
        batch,
        partner_loss: bool = True,
        gt_column: str = "abundance",
        unnormalize: bool = False,
    ) -> torch.tensor:
        graph, sample_ids = batch
        # abundance = graph.ndata[abundance_column]
        gt = graph.ndata[gt_column]
        masks = graph.ndata["mask"]
        molecule_mask = masks[self.molecule]
        num_masked_molecule = molecule_mask.sum()
        molecule_gt = gt[self.molecule].detach().clone()
        assert torch.isnan(molecule_gt[molecule_mask]).sum() == 0
        # molecule_input = abundance[self.molecule]
        # molecule_input[molecule_mask] = self.mask_value
        mol_target = molecule_gt[molecule_mask]

        partner_mask = masks[self.partner_molecule]
        num_masked_partner = partner_mask.sum()
        partner_gt = gt[self.partner_molecule].detach().clone()
        assert torch.isnan(partner_gt[partner_mask]).sum() == 0
        # partner_gt[torch.isnan(partner_gt)] = self.mask_value
        # partner_input = abundance[self.partner_molecule]
        # partner_input[partner_mask] = self.mask_value
        partner_target = partner_gt[partner_mask]

        mol_vec, partner_vec = self(batch, abundance_column="abundance")

        # print(mol_vec.mean().item(), mol_vec.std().item(), mol_vec.median().item())
        mol_pred = mol_vec[molecule_mask][:, 0]
        # print(mol_pred.mean().item(), mol_pred.std().item())
        mol_pred = mol_vec[molecule_mask][:, 0]
        partner_pred = partner_vec[partner_mask][:, 0]
        #print('pred', mol_pred.mean().item(), partner_pred.mean().item())
        #print('target', mol_target.mean().item(), partner_target.mean().item())
        # var_cap = torch.tensor([10], device=partner_vec.device, dtype=partner_vec.dtype)
        # print((torch.exp(partner_vec[partner_mask][:, 1])>10).sum().item())
        # print((torch.exp(mol_vec[molecule_mask][:, 1])>10).sum().item())
        partner_var = torch.exp(partner_vec[partner_mask][:, 1])
        # partner_vec = torch.min(partner_vec, var_cap)
        mol_var = torch.exp(mol_vec[molecule_mask][:, 1])

        molecule_loss_coefficient = 0
        partner_loss_coefficient = 1
        loss = molecule_loss_coefficient * self.loss_fn(
            mol_pred, target=mol_target, var=mol_var
        )
        self.log(
            "molecule_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1
        )
        if partner_loss and partner_loss_coefficient > 0:
            partner_loss = partner_loss_coefficient * self.loss_fn(
                partner_pred, target=partner_target, var=partner_var
            )
            loss += partner_loss
            self.log(
                "partner_loss",
                partner_loss.item(),
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        # if loss.item() > loss_thresh.item():
        #     print(f"attention exploding loss: {loss.item()}")
        loss = torch.min(loss, torch.max(partner_target.max(), mol_target.max()))

        if unnormalize and self.normalizer is not None:
            mol_pred = (
                mol_pred * self.normalizer.stds[self.molecule]["abundance"]
            ) + self.normalizer.means[self.molecule]["abundance"]
            mol_target = (
                mol_target * self.normalizer.stds[self.molecule][gt_column]
            ) + self.normalizer.means[self.molecule][gt_column]
            partner_pred = (
                partner_pred * self.normalizer.stds[self.partner_molecule]["abundance"]
            ) + self.normalizer.means[self.partner_molecule]["abundance"]
            partner_target = (
                partner_target * self.normalizer.stds[self.partner_molecule][gt_column]
            ) + self.normalizer.means[self.partner_molecule][gt_column]
            mol_var = mol_var * self.normalizer.stds[self.molecule]["abundance"]
            partner_var = (
                partner_var * self.normalizer.stds[self.partner_molecule]["abundance"]
            )

        # mol_vec = torch.min(mol_vec, var_cap)
        # loss = torch.nn.functional.mse_loss(output, inputs.mean(dim=-1, keepdim=True))
        self.log(
            "num_masked_molecule",
            int(num_masked_molecule.item()),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "num_masked_partner",
            int(num_masked_partner.item()),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        # tmp_mask = (molecule_input != self.mask_value) & (~torch.isnan(molecule_gt))
        molecule_rmse = nn.functional.mse_loss(mol_pred, target=mol_target) ** 0.5
        self.log(
            "molecule_rmse",
            molecule_rmse.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        molecule_rmse_median = ((mol_pred - mol_target) ** 2).median() ** 0.5
        self.log(
            "mol_medse",
            molecule_rmse_median.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        partner_rmse = (
            nn.functional.mse_loss(partner_pred, target=partner_target) ** 0.5
        )
        self.log(
            "partner_rmse",
            partner_rmse.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        partner_rmse_median = (((partner_pred - partner_target) ** 2).mean(axis=0) ** 0.5).median()
        self.log(
            "partner_medse",
            partner_rmse_median.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        molecule_pearsonr = torch.corrcoef(torch.stack([mol_pred, mol_target], dim=0))[
            0, 1
        ]
        self.log(
            "molecule_pearsonr",
            molecule_pearsonr.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        partner_pearsonr = torch.corrcoef(
            torch.stack([partner_pred, partner_target], dim=0)
        )[0, 1]
        self.log(
            "partner_pearsonr",
            partner_pearsonr.item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "var mean",
            mol_var.mean().item(),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(
            batch, partner_loss=True, gt_column="gt", unnormalize=True
        )
        self.log("val_loss", loss.item(), batch_size=1)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        mol_vec, partner_vec = self(batch)
        mol_pred, mol_var = mol_vec[:, :, 0], torch.exp(mol_vec[:, :, 1])
        partner_pred, partner_var = partner_vec[:, :, 0], torch.exp(
            partner_vec[:, :, 1]
        )
        return mol_pred, mol_var, partner_pred, partner_var

    def configure_optimizers(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def impute_sequ_embedding_gnn(
    dataset: Dataset,
    molecule: str,
    column: str,
    mapping: str,
    partner_column: str,
    embedding_molecule_column: str,
    embedding_partner_column: str,
    molecule_result_column: Optional[str] = None,
    molecule_uncertainty_column: Optional[str] = None,
    partner_result_column: Optional[str] = None,
    partner_uncertainty_column: Optional[str] = None,
    gt_column: Optional[str] = None,
    partner_gt_column: Optional[str] = None,
    max_epochs: int = 5000,
    training_fraction: float = 0.25,
    train_sample_wise: bool = False,
    log_every_n_steps: Optional[int] = None,
    early_stopping_patience: int = 7,
    logger: Optional[Logger] = None,
    epoch_size: int = 30,
    masking_seed: Optional[int] = None,
    missing_substitute_value: int = -2,
    num_workers: int = 0,
    embedding_shrink_dim: int = 16,
) -> pd.Series:
    """Impute missing values using a homogenous graph neural network applied on the molecule graph created from two molecule types like proteins and their assigned peptides.

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
        masking_seed (Optional[int], optional): If given this seed is used to seed the random generator for randomly masking molecule values during training. Defaults to None.
        missing_substitute_value (float, optional): Value to replace missing or masked values with. Defaults to -3.
    Returns:
        pd.Series: the imputed values.
    """
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    # train_sample_wise = False
    if log_every_n_steps is None:
        if train_sample_wise:
            log_every_n_steps = dataset.num_samples
        else:
            log_every_n_steps = 1

    # #determining the masking fraction to resemble the missing fraction of partner molecule in the dataset
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
    # missing_fraction = mapped_missing[partner_column].isna().sum() / mapped_missing.shape[0]
    # overall_missing_fraction = mapped[partner_column].isna().sum() / mapped.shape[0]
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
    # #masking_fraction = training_fraction
    # print(f"masking_fraction: {masking_fraction}")

    ds = dataset.copy(
        columns={
            molecule: [column] + ([gt_column] if gt_column is not None else []),
            partner_molecule: [partner_column]
            + ([partner_gt_column] if partner_gt_column is not None else []),
        }
    )
    column_name_map = {
        molecule: {column: "abundance"},
        partner_molecule: {partner_column: "abundance"},
    }
    if gt_column is not None:
        column_name_map[molecule][gt_column] = "gt"
    if partner_gt_column is not None:
        column_name_map[partner_molecule][partner_gt_column] = "gt"
    ds.rename_columns(
        columns=column_name_map,
        inplace=True,
    )

    normalizer = Standardizer(columns=["abundance", "gt"])
    normalizer.standardize(dataset=ds, inplace=True)

    if gt_column is not None and partner_gt_column is not None:
        validation_ids = ds.values[molecule]["abundance"]
        validation_ids = validation_ids[validation_ids.isna()]
        gt_ids = ds.values[molecule]["gt"]
        gt_ids = gt_ids[~gt_ids.isna()].index
        validation_ids = validation_ids[validation_ids.index.isin(gt_ids)].index
        partner_validation_ids = ds.values[partner_molecule]["abundance"]
        partner_validation_ids = partner_validation_ids[partner_validation_ids.isna()]
        partner_gt_ids = ds.values[partner_molecule]["gt"]
        partner_gt_ids = partner_gt_ids[~partner_gt_ids.isna()].index
        partner_validation_ids = partner_validation_ids[
            partner_validation_ids.index.isin(partner_gt_ids)
        ].index
        validation_set = MaskedDataset.from_ids(
            dataset=ds,
            mask_ids={
                molecule: validation_ids,
                partner_molecule: partner_validation_ids,
            },
        )

    # ATTENTION ONLY FOR DEBUGGING
    #ds.values[molecule]["abundance"] = ds.values[molecule]["gt"]
    #ds.values[partner_molecule]["abundance"] = ds.values[partner_molecule]["gt"]

    # mapping_df = ds.mappings[mapping].df
    # partner_mask_ids = ds.values[partner_molecule]["abundance"]
    # Make sure to only mask partner molecules connected to at least one molecule
    # partner_mask_ids = partner_mask_ids[partner_mask_ids.index.get_level_values('id').isin(mapping_df.index.get_level_values(partner_molecule).unique())]
    # partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
    # if masking_seed is None:
    #     masking_seed = random.randint(0, 1000000000)
    # print(f"masking seed: {masking_seed}")
    # non_missing_mols = vals[~vals.isna()].index
    mapping_df = ds.mappings[mapping].df
    partner_mask_ids = ds.values[partner_molecule]["abundance"]
    # Make sure to only mask partner molecules connected to at least one molecule
    partner_mask_ids = partner_mask_ids[
        partner_mask_ids.index.get_level_values("id").isin(
            mapping_df.index.get_level_values(partner_molecule).unique()
        )
    ]
    partner_mask_ids = partner_mask_ids[~partner_mask_ids.isna()]
    non_missing_per_missing_mol = (
        ds.get_mapped(molecule, mapping=mapping, partner_columns=["abundance"])
        .groupby(molecule)
        .abundance.count()
    )
    #non_missing_per_missing_mol.index.set_names(molecule, inplace=True)
    #non_missing_per_missing_mol = non_missing_per_missing_mol.swaplevel()
    vals = ds.values[molecule]["abundance"]
    missing_mols = vals[vals.isna()].index
    non_missing_mols = vals[~vals.isna()]
    non_missing_per_missing_mol = non_missing_per_missing_mol[
        non_missing_per_missing_mol.index.isin(missing_mols.get_level_values('id').unique())
    ].value_counts(normalize=True)
    partner_missingness = ds.get_wf(molecule=partner_molecule, column="abundance")
    partner_missingness = (
        partner_missingness.isna().sum(axis=1) / partner_missingness.shape[1]
    )

    # mask_template = mapped.index.droplevel(molecule)
    # mask_template.set_names('id', level=partner_molecule, inplace=True)
    # mapped_index_map = pd.Series(np.arange(partner_mask_ids.shape[0]), partner_mask_ids.index)
    # mapped_index_map = mapped_index_map.loc[mapped.index.droplevel(molecule).set_names('id', level=partner_molecule)]

    def find_mask_ids(in_ds: Dataset):
        rng = np.random.default_rng()
        epoch_masking_fraction = rng.uniform(
            0.1 * training_fraction, 1.5 * training_fraction
        ) / 2
        mcar_ids = partner_mask_ids.sample(  # [~partner_mask_ids.index.isin(partner_validation_ids)]
            frac=epoch_masking_fraction
        ).index
        mnar_ids = in_ds.molecules[partner_molecule].sample(frac=epoch_masking_fraction).index
        mnar_ids = partner_mask_ids[partner_mask_ids.index.get_level_values('id').isin(mnar_ids)].index
        partner_ids = mcar_ids.union(mnar_ids)
        # cp = in_ds.copy(copy_molecule_set=True)
        # partner_vals = cp.values[partner_molecule]["abundance"]
        # partner_vals.loc[partner_ids] = np.nan
        # cp.values[partner_molecule]["abundance"] = partner_vals
        # #missingness = cp.wf(molecule=partner_molecule, column="abundance")
        # #missingness = missingness.isna().sum(axis=1) / missingness.shape[1]
        # mapped = cp.get_mapped(molecule, mapping=mapping, partner_columns=["abundance"])
        # #mapped['missingness'] = missingness.loc[mapped.index.get_level_values(partner_molecule)].values
        # groups = mapped.groupby(molecule)
        # candidates = groups.abundance.count()
        # candidates = candidates[
        #     candidates.index.isin(non_missing_mols.index.get_level_values('id').unique())
        # ]
        # candidates = candidates.map(non_missing_per_missing_mol)
        # candidates[candidates.isna()] = 0
        # molecule_ids = non_missing_mols.sample(
        #     frac=2 * epoch_masking_fraction, weights=candidates.loc[non_missing_mols.index.get_level_values('id')].values
        # ).index
        molecule_ids = non_missing_mols.index
        return molecule_ids, partner_ids

    def masking_fn(in_ds: Dataset):
        molecule_ids, partner_ids = find_mask_ids(in_ds)
        return MaskedDataset.from_ids(
            dataset=in_ds,
            mask_ids={
                molecule: molecule_ids,
                partner_molecule: partner_ids,
            },
            # hidden_ids={molecule: validation_ids}#, partner_molecule: partner_validation_ids},
        )

    mask_ds = MaskedDatasetGenerator(
        datasets=[ds],
        generator_fn=masking_fn,
        sample_wise=train_sample_wise,
        epoch_size_multiplier=epoch_size,
        shuffle_samplewise_samples=True,
    )

    collator = GraphCollator()

    sample_to_id_map = {s: i for i, s in enumerate(ds.sample_names)}

    def collate(
        mds: List[Tuple[MaskedDataset, List[str]]],
        feature_columns: List[str] = ["abundance"],
    ):
        assert len(mds) == 1
        res = []
        sample_ids = []
        for md, samples in mds:
            if samples is None:
                samples = ds.sample_names
            graph = md.to_dgl_graph(
                feature_columns={
                    molecule: feature_columns,
                    partner_molecule: feature_columns,
                },
                molecule_columns={
                    molecule: embedding_molecule_column,
                    partner_molecule: embedding_partner_column,
                },
                mappings=[mapping],
                mapping_directions={mapping: (partner_molecule, molecule)},
                make_bidirectional=True,
                samples=samples,
            )
            res.append(graph)
            sample_ids.append([sample_to_id_map[s] for s in samples])
        return collator.collate(res), np.array(sample_ids)

    train_dl = DataLoader(
        mask_ds, batch_size=1, collate_fn=collate, num_workers=num_workers
    )
    if gt_column is not None and partner_gt_column is not None:
        if train_sample_wise:
            validation_dl = DataLoader(
                [(validation_set, [s]) for s in ds.sample_names],
                batch_size=1,
                collate_fn=functools.partial(
                    collate, feature_columns=["abundance", "gt"]
                ),
            )
        else:
            validation_dl = DataLoader(
                [(validation_set, None)],
                batch_size=1,
                collate_fn=functools.partial(
                    collate, feature_columns=["abundance", "gt"]
                ),
            )
    else:
        validation_dl = None

    # if molecule_coefficient is None:
    #     molecule_coefficient = 1
    # if partner_coefficient is None:
    #     partner_coefficient = ((~ds.values[molecule]["abundance"].isna()).sum() * training_fraction) / ((~ds.values[partner_molecule]['abundance'].isna()).sum() * masking_fraction)
    # print(f"molecule_coefficient: {molecule_coefficient}, partner_coefficient: {partner_coefficient}")
    num_samples = dataset.num_samples
    in_dim = 1 if train_sample_wise else num_samples
    if train_sample_wise:
        layer_dim_dict = {
            "layers": [16],
            "gat_heads": 64,
            "gat_dim": 64,
        }
    else:
        layer_dim_dict = {
            "layers": [num_samples * num_samples, 2 * num_samples, 2 * num_samples],
            "gat_heads": num_samples,
            "gat_dim": 4*num_samples,
        }
    model = ImputationModule(
        molecule=molecule,
        partner_molecule=partner_molecule,
        mapping=mapping,
        in_dim=in_dim,
        # molecule_loss_coefficent=molecule_coefficient,
        # partner_loss_coefficent=partner_coefficient,
        dropout=0.10,  # 30,
        lr=0.005,
        num_samples=ds.num_samples,
        embedding_in_dim=1024,
        embedding_shrink_dim=embedding_shrink_dim,
        mask_value=missing_substitute_value,
        embedding_molecule_key=embedding_molecule_column,
        embedding_partner_key=embedding_partner_column,
        normalizer=normalizer,
        use_sample_embeddings=train_sample_wise,
        **layer_dim_dict
    )
    if logger is None:
        logger = ConsoleLogger()
    trainer = L.Trainer(
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
        enable_checkpointing=False,
        callbacks=[
            TrainingEarlyStopping(
                monitor="train_loss", mode="min", patience=early_stopping_patience
            )
        ],
        gradient_clip_val=1,
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
    if train_sample_wise:
        sample_names = [[s] for s in ds.sample_names]
        pred_dl = DataLoader(
            [(missing_mds, s) for s in sample_names], batch_size=1, collate_fn=collate
        )
    else:
        sample_names = [None]
        pred_dl = DataLoader([(missing_mds, None)], batch_size=1, collate_fn=collate)
    for (pred_mol, uncertainty_mol, pred_partner, uncertainty_partner), s in zip(
        trainer.predict(model=model, dataloaders=pred_dl, ckpt_path=None), sample_names
    ):
        pred_mol, uncertainty_mol = pred_mol.numpy(), uncertainty_mol.numpy()
        print(pred_mol.mean(), pred_mol.std())
        pred_partner, uncertainty_partner = (
            pred_partner.numpy(),
            uncertainty_partner.numpy(),
        )

        missing_mds.set_samples_value_matrix(
            matrix=pred_mol,
            molecule=molecule,
            column="abundance",
            only_set_masked=True,
            samples=s,
        )
        missing_mds.set_samples_value_matrix(
            matrix=pred_partner,
            molecule=partner_molecule,
            column="abundance",
            only_set_masked=True,
            samples=s,
        )

        missing_mds.set_samples_value_matrix(
            matrix=uncertainty_mol * normalizer.stds[molecule]["abundance"],
            molecule=molecule,
            column="uncertainty",
            only_set_masked=True,
            samples=s,
        )
        missing_mds.set_samples_value_matrix(
            matrix=uncertainty_partner * normalizer.stds[partner_molecule]["abundance"],
            molecule=partner_molecule,
            column="uncertainty",
            only_set_masked=True,
            samples=s,
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
