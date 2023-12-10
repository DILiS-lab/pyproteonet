from typing import Optional, List

import numpy as np
import torch
import dgl
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyproteonet.lightning import ConsoleLogger
from dgl.dataloading import GraphCollator

from ...data.dataset import Dataset
from ...masking.train_eval import (
    train_eval_protein_and_mapped,
    train_eval_full_protein_and_mapped,
    train_eval_full_protein_and_mapped_backup,
    train_eval_full_molecule,
    train_eval_full_molecule_some_mapped
)
from ...normalization.dnn_normalizer import DnnNormalizer
from ...dgl.collate import (
    masked_dataset_to_homogeneous_graph,
    masked_heterograph_to_homogeneous,
)
from ...lightning.uncertainty_gat_node_imputer import UncertaintyGatNodeImputer
from ...masking.missing_values import mask_missing


def impute_all_sample_gnn(
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
):
    molecule, mapping, peptide_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )

    in_dataset = dataset.copy(
        columns={
            molecule: [column]
            if protein_gt_column is None
            else [column, protein_gt_column],
            peptide_molecule: [partner_column],
        }
    )
    in_dataset.rename_values(
        columns={column: "abundance"},
        molecules=[molecule],
        inplace=True,
    )
    in_dataset.rename_values(
        columns={partner_column: "abundance"},
        molecules=[peptide_molecule],
        inplace=True,
    )
    if protein_gt_column is not None:
        in_dataset.rename_values(
            columns={protein_gt_column: "abundance_gt"},
            molecules=[molecule],
            inplace=True,
        )
        in_dataset.values[peptide_molecule]["abundance_gt"] = in_dataset.values[
            peptide_molecule
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
    train_ds, eval_ds = train_eval_full_protein_and_mapped(
        dataset=in_dataset,
        molecule=molecule,
        column="abundance",
        partner_column="abundance",
        mapping=mapping,
        validation_fraction=validation_fraction,
        training_fraction=training_fraction,
        partner_masking_fraction=partner_masking_fraction,
    )

    collator = GraphCollator()
    collate_fn = lambda masked_datasets: collator.collate(
        masked_dataset_to_homogeneous_graph(
            masked_datasets=masked_datasets,
            mappings=[mapping],
            target="abundance",
            features=[],
        )
    )
    train_dl = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)
    eval_dl = DataLoader([eval_ds], batch_size=1, collate_fn=collate_fn)
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
