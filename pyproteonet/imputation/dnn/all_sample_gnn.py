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
from ...masking.train_eval import train_eval_protein_and_mapped
from ...normalization.dnn_normalizer import DnnNormalizer
from ...dgl.collate import (
    masked_dataset_to_homogeneous_graph,
    masked_heterograph_to_homogeneous,
)
from ...lightning.gat_node_imputer import GatNodeImputer
from ...lightning.uncertainty_gat_node_imputer import UncertaintyGatNodeImputer
from ...masking.masking import mask_missing


def impute_all_sample_gnn(
    dataset: Dataset,
    protein_abundance_column: str,
    peptide_abundance_column: str,
    validation_fraction=0.1,
    training_fraction=0.5,
    peptide_masking_fraction=0.5,
    protein_molecule="protein",
    peptide_molecule="peptide",
    mapping: Optional[str] = None,
    result_column: Optional[str] = None,
    max_epochs: int = 10000,
    validation_frequency: int = 50,
    early_stopping_patience: int = 5,
    missing_substitute_value: float = -3,
    use_gatv2: bool = True,
    train_on_mapped: bool = True,
    uncertainty_column: Optional[str] = None,
):
    if mapping is None:
        protein_molecule, mapping, peptide_molecule = dataset.infer_mapping(
            molecule=protein_molecule, mapping=peptide_molecule
        )

    in_dataset = dataset.copy(
        columns={
            protein_molecule: [protein_abundance_column],
            peptide_molecule: [peptide_abundance_column],
        }
    )
    in_dataset.rename_columns(
        columns={protein_abundance_column: "abundance"},
        molecules=[protein_molecule],
        inplace=True,
    )
    in_dataset.rename_columns(
        columns={peptide_abundance_column: "abundance"},
        molecules=[peptide_molecule],
        inplace=True,
    )
    normalizer = DnnNormalizer(columns=["abundance"])
    normalizer.normalize(dataset=in_dataset, inplace=True)

    train_ds, eval_ds = train_eval_protein_and_mapped(
        dataset=in_dataset,
        protein_abundance_column="abundance",
        peptide_abundance_column="abundance",
        validation_fraction=validation_fraction,
        training_fraction=training_fraction,
        peptide_masking_fraction=peptide_masking_fraction,
        protein_molecule=protein_molecule,
        mapping=peptide_molecule,
        train_mapped=train_on_mapped,
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

    num_samples = len(in_dataset.sample_names)
    heads = [num_samples, num_samples, num_samples]  # [4*num_samples]
    dimensions = [4, 2, 2]  # [1]#, num_samples, num_samples]
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
        initial_dense_layers=[8 * num_samples, 2 * num_samples],
        lr=0.001,
    )

    trainer = Trainer(
        logger=ConsoleLogger(),
        max_epochs=max_epochs,
        enable_checkpointing=False,
        check_val_every_n_epoch=validation_frequency,
        log_every_n_steps=50,
        callbacks=[
            EarlyStopping(
                monitor="validation_loss", mode="min", patience=early_stopping_patience
            )
        ],
    )
    module.uncertainty_loss = False
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    if uncertainty_column is not None:
        module.uncertainty_loss = True
        module.lr = 0.0001
        trainer = Trainer(
            logger=ConsoleLogger(),
            max_epochs=max_epochs,
            enable_checkpointing=False,
            check_val_every_n_epoch=validation_frequency,
            log_every_n_steps=50,
            callbacks=[
                EarlyStopping(
                    monitor="validation_loss",
                    mode="min",
                    patience=early_stopping_patience,
                )
            ],
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=eval_dl)

    predict_ds = mask_missing(
        dataset=in_dataset, molecule=protein_molecule, column="abundance"
    )
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
    protein_index = ntypes.index(protein_molecule)
    protein_mask = (predict_graph.ndata[dgl.NTYPE] == protein_index).type(torch.bool)
    res = trainer.predict(
        module, DataLoader([predict_graph], batch_size=1, collate_fn=collator.collate)
    )[0]
    prot_res = res[protein_mask]
    masked_proteins = predict_graph.ndata["mask"][protein_mask].type(torch.bool)
    protein_ids = predict_graph.ndata[dgl.NID][protein_mask].type(torch.int64)
    prot_res = prot_res[protein_ids, :]
    masked_proteins = masked_proteins[protein_ids, :]
    mat_pd = in_dataset.get_samples_value_matrix(
        molecule=protein_molecule, column="abundance"
    )
    mat = mat_pd.to_numpy()
    mat[masked_proteins.numpy()] = prot_res[masked_proteins].numpy()[:, 0]
    # mat[:, :] = prot_res[:, :].numpy()
    mat_pd.loc[:, :] = mat
    in_dataset.set_samples_value_matrix(
        matrix=mat_pd, molecule=protein_molecule, column="abundance"
    )
    normalizer.unnormalize(dataset=in_dataset, inplace=True)
    vals = dataset.values[protein_molecule][protein_abundance_column]
    res_vals = in_dataset.values[protein_molecule]["abundance"]
    vals.loc[vals.isna(), :] = res_vals.loc[vals.isna(), :]
    if result_column is not None:
        dataset.values[protein_molecule][result_column] = vals
    if uncertainty_column is not None:
        mat[:, :] = np.nan
        mat[masked_proteins.numpy()] = prot_res[masked_proteins].numpy()[:, 1]
        mat_pd.loc[:, :] = mat
        dataset.set_samples_value_matrix(
            matrix=mat_pd, molecule=protein_molecule, column=uncertainty_column
        )
    return vals
