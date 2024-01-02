from typing import Tuple
import random

import numpy as np
import pandas as pd

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset
from .masked_dataset_generator import MaskedDatasetGenerator
from .missing_values import mask_non_missing


def train_eval_protein_and_mapped(
    dataset: Dataset,
    protein_abundance_column: str,
    peptide_abundance_column: str,
    validation_fraction=0.2,
    training_fraction=0.1,
    peptide_masking_fraction=0.5,
    protein_molecule="protein",
    mapping="peptide",
    train_mapped: bool = False,
) -> Tuple[MaskedDatasetGenerator, MaskedDataset]:
    protein_molecule, mapping, peptide_molecule = dataset.infer_mapping(
        molecule=protein_molecule, mapping=mapping
    )
    validation_proteins = (
        dataset.molecules[protein_molecule].sample(frac=validation_fraction).index
    )
    eval_protein_mask = mask_non_missing(
        dataset=dataset,
        molecule=protein_molecule,
        column=protein_abundance_column,
        ids=validation_proteins,
        frac=None,
    )
    eval_protein_mask = eval_protein_mask.masks[protein_molecule]
    mapping = dataset.mappings[mapping].df
    if train_mapped:
        validation_peptides = (
            mapping[
                mapping.index.get_level_values(protein_molecule).isin(
                    validation_proteins
                )
            ]
            .sample(frac=validation_fraction)
            .index.get_level_values(1)
        )
        eval_peptide_mask = mask_non_missing(
            dataset=dataset,
            molecule=peptide_molecule,
            column=peptide_abundance_column,
            ids=validation_peptides,
            frac=None,
        )
        eval_peptide_mask = eval_peptide_mask.masks[peptide_molecule]
        eval_dataset = MaskedDataset(
            dataset=dataset,
            masks={
                protein_molecule: eval_protein_mask,
                peptide_molecule: eval_peptide_mask,
            },
        )
    else:
        eval_dataset = MaskedDataset(
            dataset=dataset, masks={protein_molecule: eval_protein_mask}
        )

    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=protein_molecule, column=protein_abundance_column
        )
        if isinstance(training_fraction, tuple):
            train_frac = random.uniform(training_fraction[0], training_fraction[1])
        else:
            train_frac = training_fraction
        if isinstance(peptide_masking_fraction, tuple):
            pep_frac = random.uniform(
                peptide_masking_fraction[0], peptide_masking_fraction[1]
            )
        else:
            pep_frac = peptide_masking_fraction
        relevant_peptides = dict()
        for sample in input_dataset.sample_names:
            ids = prot_mask[sample].isna()
            ids = ids[~ids]
            ids = (
                ids[~ids.index.isin(validation_proteins)].sample(frac=train_frac).index
            )
            prot_mask[sample] = False
            prot_mask.loc[ids, sample] = True
            relevant_peptides[sample] = mapping[
                mapping.index.get_level_values(protein_molecule).isin(ids)
            ].index.get_level_values(1)
        pep_mask = input_dataset.get_samples_value_matrix(
            molecule=peptide_molecule, column=peptide_abundance_column
        )
        for sample in input_dataset.sample_names:
            ids = pep_mask[sample].isna()
            ids = ids[~ids]
            ids = ids[ids.index.isin(relevant_peptides[sample])]
            if train_mapped:
                ids = ids[~ids.index.isin(validation_peptides)]
            ids = ids.sample(frac=pep_frac).index
            pep_mask[sample] = False
            pep_mask.loc[ids, sample] = True
        if train_mapped:
            return MaskedDataset(
                dataset=dataset,
                masks={protein_molecule: prot_mask, peptide_molecule: pep_mask},
                hidden={
                    protein_molecule: eval_protein_mask,
                    peptide_molecule: eval_peptide_mask,
                },
            )
        else:
            return MaskedDataset(
                dataset=dataset,
                masks={protein_molecule: prot_mask},
                hidden={
                    protein_molecule: eval_protein_mask,
                    peptide_molecule: pep_mask,
                },
            )

    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset


def train_eval_full_protein_and_mapped_backup(
    dataset: Dataset,
    protein_abundance_column: str,
    peptide_abundance_column: str,
    validation_fraction=0.2,
    training_fraction=0.1,
    peptide_masking_fraction=0.5,
    protein_molecule="protein",
    mapping="peptide",
    train_mapped: bool = False,
    max_mnar_quantile=0,
) -> Tuple[MaskedDatasetGenerator, MaskedDataset]:
    protein_molecule, mapping, peptide_molecule = dataset.infer_mapping(
        molecule=protein_molecule, mapping=mapping
    )
    validation_proteins = (
        dataset.molecules[protein_molecule].sample(frac=validation_fraction).index
    )
    eval_protein_mask = mask_non_missing(
        dataset=dataset,
        molecule=protein_molecule,
        column=protein_abundance_column,
        ids=validation_proteins,
        frac=None,
    )
    eval_protein_mask = eval_protein_mask.masks[protein_molecule]
    mapping = dataset.mappings[mapping].df
    if train_mapped:
        validation_peptides = (
            mapping[mapping.index.get_level_values(0).isin(validation_proteins)]
            .sample(frac=validation_fraction)
            .index.get_level_values(1)
        )
        eval_peptide_mask = mask_non_missing(
            dataset=dataset,
            molecule=peptide_molecule,
            column=peptide_abundance_column,
            ids=validation_peptides,
            frac=None,
        )
        eval_peptide_mask = eval_peptide_mask.masks[peptide_molecule]
        eval_dataset = MaskedDataset(
            dataset=dataset,
            masks={
                protein_molecule: eval_protein_mask,
                peptide_molecule: eval_peptide_mask,
            },
        )
    else:
        eval_dataset = MaskedDataset(
            dataset=dataset, masks={protein_molecule: eval_protein_mask}
        )

    available_prots = dataset.get_samples_value_matrix(
        molecule=protein_molecule, column=protein_abundance_column
    )
    available_prots = (~available_prots.isna()).sum(axis=1)
    available_prots = available_prots[
        (available_prots > 0) & ~available_prots.index.isin(validation_proteins)
    ]

    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=protein_molecule, column=protein_abundance_column
        )
        if isinstance(training_fraction, tuple):
            train_frac = random.uniform(training_fraction[0], training_fraction[1])
        else:
            train_frac = training_fraction
        if isinstance(peptide_masking_fraction, tuple):
            pep_frac = random.uniform(
                peptide_masking_fraction[0], peptide_masking_fraction[1]
            )
        else:
            pep_frac = peptide_masking_fraction
        masked_prots = available_prots.sample(frac=train_frac).index
        relevant_peptides = mapping[
            mapping.index.get_level_values(0).isin(masked_prots)
        ].index.get_level_values(1)
        for sample_name, sample in input_dataset.samples_dict.items():
            ids = sample.values[protein_molecule].loc[
                masked_prots, protein_abundance_column
            ]
            ids = ids[~ids.isna()].index
            prot_mask[sample_name] = False
            prot_mask.loc[ids, sample_name] = True
        pep_mask = input_dataset.get_samples_value_matrix(
            molecule=peptide_molecule, column=peptide_abundance_column
        )
        for sample_name in input_dataset.sample_names:
            ids = pep_mask[sample_name].isna()
            ids = ids[~ids]
            ids = ids[ids.index.isin(relevant_peptides)]
            if train_mapped:
                ids = ids[~ids.index.isin(validation_peptides)]
            ids = ids.sample(frac=pep_frac).index
            pep_mask[sample_name] = False
            pep_mask.loc[ids, sample_name] = True
        if train_mapped:
            return MaskedDataset(
                dataset=dataset,
                masks={protein_molecule: prot_mask, peptide_molecule: pep_mask},
                hidden={
                    protein_molecule: eval_protein_mask,
                    peptide_molecule: eval_peptide_mask,
                },
            )
        else:
            return MaskedDataset(
                dataset=dataset,
                masks={protein_molecule: prot_mask},
                hidden={
                    protein_molecule: eval_protein_mask,
                    peptide_molecule: pep_mask,
                },
            )

    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset


def train_eval_full_protein_and_mapped(
    dataset: Dataset,
    molecule: str,
    column: str,
    partner_column: str,
    validation_fraction=0.2,
    training_fraction=0.4,
    partner_masking_fraction=0.4,
    mapping="peptide",
    train_mapped: bool = False,
    max_mnar_quantile=0,
) -> Tuple[MaskedDatasetGenerator, MaskedDataset]:
    molecule, mapping, peptide_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    validation_proteins = (
        dataset.molecules[molecule].sample(frac=validation_fraction).index
    )
    eval_protein_mask = mask_non_missing(
        dataset=dataset,
        molecule=molecule,
        column=column,
        ids=validation_proteins,
        frac=None,
    )
    eval_protein_mask = eval_protein_mask.masks[molecule]
    mapping = dataset.mappings[mapping].df
    if train_mapped:
        validation_peptides = (
            mapping[
                mapping.index.get_level_values(molecule).isin(
                    validation_proteins
                )
            ]
            .sample(
                frac=validation_fraction * (validation_fraction / training_fraction)
            )
            .index.get_level_values(peptide_molecule)
        )
        eval_peptide_mask = mask_non_missing(
            dataset=dataset,
            molecule=peptide_molecule,
            column=partner_column,
            ids=validation_peptides,
            frac=None,
        )
        eval_peptide_mask = eval_peptide_mask.masks[peptide_molecule]
        eval_dataset = MaskedDataset(
            dataset=dataset,
            masks={
                molecule: eval_protein_mask,
                peptide_molecule: eval_peptide_mask,
            },
        )
    else:
        eval_dataset = MaskedDataset(
            dataset=dataset, masks={molecule: eval_protein_mask}
        )

    prot_mat = dataset.get_samples_value_matrix(
        molecule=molecule, column=column
    )
    available_prots = (~prot_mat.isna()).sum(axis=1)
    available_prots = available_prots[
        (available_prots > 0) & ~available_prots.index.isin(validation_proteins)
    ]
    rng = np.random.default_rng()

    missing_prots = dict()
    for sample_name in dataset.sample_names:
        missing = prot_mat[sample_name]
        missing = missing[missing.isna()]
        missing_prots[sample_name] = missing.index

    
    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=molecule, column=column
        )
        if isinstance(training_fraction, tuple):
            train_frac = random.uniform(training_fraction[0], training_fraction[1])
        else:
            train_frac = training_fraction
        if isinstance(partner_masking_fraction, tuple):
            pep_frac = random.uniform(
                partner_masking_fraction[0], partner_masking_fraction[1]
            )
        else:
            pep_frac = partner_masking_fraction

        # num_bins = 10
        # prot_mean = prot_mask.loc[available_prots.index].mean(axis=1)
        # prot_bin = pd.cut(prot_mean, bins=num_bins, labels=False)
        # masked_prots = prot_mean.groupby(prot_bin).apply(lambda x: x.sample(frac=train_frac)).index.get_level_values(1)
        masked_prots = available_prots.sample(frac=train_frac).index
        relevant_peptides = mapping[
            mapping.index.get_level_values(molecule).isin(masked_prots)
        ].index.get_level_values(peptide_molecule)
        for sample_name, sample in input_dataset.samples_dict.items():
            ids = sample.values[molecule].loc[
                masked_prots, column
            ]
            ids = ids[~ids.isna()].index
            prot_mask[sample_name] = False
            prot_mask.loc[ids, sample_name] = True
        pep_mask = input_dataset.get_samples_value_matrix(
            molecule=peptide_molecule, column=partner_column
        )
        pep_vals = pep_mask.copy()
        mnar_quantile = rng.uniform(0, max_mnar_quantile)
        mnar_quantile_val = np.nanquantile(pep_mask.values, mnar_quantile)
        mnar_std = mnar_quantile_val - np.nanquantile(
            pep_mask.values, mnar_quantile - mnar_quantile / 4
        )
        for sample_name in input_dataset.sample_names:
            missing_prot_peps = mapping[
                mapping.index.get_level_values(molecule).isin(missing_prots[sample_name])
            ].index.get_level_values(peptide_molecule)
            ids = pep_mask[sample_name].isna()
            ids = ids[~ids]
            #ids = ids[ids.index.isin(missing_prot_peps)]
            ids = ids[ids.index.isin(relevant_peptides)]
            if train_mapped:
                ids = ids[~ids.index.isin(validation_peptides)]
            ids_mcar = ids.sample(frac=pep_frac).index
            pep_mask[sample_name] = False
            pep_mask.loc[ids_mcar, sample_name] = True
            if max_mnar_quantile > 0:
                thresh = rng.normal(
                    loc=mnar_quantile_val, scale=mnar_std, size=ids.shape[0]
                )
                vals = pep_vals.loc[ids.index, sample_name]
                ids_mnar = vals.loc[vals < thresh].index
                pep_mask.loc[ids_mnar, sample_name] = True
        # print("protein", prot_mask.sum().sum())
        # print("peptide", pep_mask.sum().sum())
        if train_mapped:
            return MaskedDataset(
                dataset=dataset,
                masks={molecule: prot_mask, peptide_molecule: pep_mask},
                hidden={
                    molecule: eval_protein_mask,
                    peptide_molecule: eval_peptide_mask,
                },
            )
        else:
            return MaskedDataset(
                dataset=dataset,
                masks={molecule: prot_mask},
                hidden={
                    molecule: eval_protein_mask,
                    peptide_molecule: pep_mask,
                },
            )

    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset


def train_eval_full_molecule(
    dataset: Dataset,
    molecule: str,
    column: str,
    training_fraction=0.4,
    validation_fraction=0.2,
) -> Tuple[MaskedDatasetGenerator, MaskedDataset]:
    validation_proteins = (
        dataset.molecules[molecule].sample(frac=validation_fraction).index
    )
    eval_protein_mask = mask_non_missing(
        dataset=dataset,
        molecule=molecule,
        column=column,
        ids=validation_proteins,
        frac=None,
    )
    eval_protein_mask = eval_protein_mask.masks[molecule]
    eval_dataset = MaskedDataset(
        dataset=dataset, masks={molecule: eval_protein_mask}
    )

    prot_mat = dataset.get_samples_value_matrix(
        molecule=molecule, column=column
    )
    available_prots = (~prot_mat.isna()).sum(axis=1)
    available_prots = available_prots[
        (available_prots > 0) & ~available_prots.index.isin(validation_proteins)
    ]
    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=molecule, column=column
        )
        if isinstance(training_fraction, tuple):
            train_frac = random.uniform(training_fraction[0], training_fraction[1])
        else:
            train_frac = training_fraction

       
        masked_prots = available_prots.sample(frac=train_frac).index
        for sample_name, sample in input_dataset.samples_dict.items():
            ids = sample.values[molecule].loc[
                masked_prots, column
            ]
            ids = ids[~ids.isna()].index
            prot_mask[sample_name] = False
            prot_mask.loc[ids, sample_name] = True
        return MaskedDataset(
            dataset=dataset,
            masks={molecule: prot_mask},
            hidden={
                molecule: eval_protein_mask,
            },
        )

    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset


def train_eval_full_molecule_some_mapped(
    dataset: Dataset,
    molecule: str,
    column: str,
    partner_column: str,
    mapping: str,
    training_fraction=0.4,
    validation_fraction=0.2,
    partner_hide_fraction=0.1
) -> Tuple[MaskedDatasetGenerator, MaskedDataset]:
    print("SDFSDF")
    molecule, mapping, partner_molecule = dataset.infer_mapping(
        molecule=molecule, mapping=mapping
    )
    mapping = dataset.mappings[mapping].df
    validation_proteins = (
        dataset.molecules[molecule].sample(frac=validation_fraction).index
    )
    eval_protein_mask = mask_non_missing(
        dataset=dataset,
        molecule=molecule,
        column=column,
        ids=validation_proteins,
        frac=None,
    )
    eval_protein_mask = eval_protein_mask.masks[molecule]
    eval_dataset = MaskedDataset(
        dataset=dataset, masks={molecule: eval_protein_mask}
    )

    prot_mat = dataset.get_samples_value_matrix(
        molecule=molecule, column=column
    )
    available_prots = (~prot_mat.isna()).sum(axis=1)
    available_prots = available_prots[
        (available_prots > 0) & ~available_prots.index.isin(validation_proteins)
    ]
    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=molecule, column=column
        )
        if isinstance(training_fraction, tuple):
            train_frac = random.uniform(training_fraction[0], training_fraction[1])
        else:
            train_frac = training_fraction

       
        masked_prots = available_prots.sample(frac=train_frac).index
        partner_mask = input_dataset.get_samples_value_matrix(
            molecule=partner_molecule, column=partner_column
        )
        for sample_name, sample in input_dataset.samples_dict.items():
            ids = sample.values[molecule].loc[
                masked_prots, column
            ]
            ids = ids[~ids.isna()]
            prot_mask[sample_name] = False
            prot_mask.loc[ids.index, sample_name] = True
            partner_mask_ids = ids.sample(frac=partner_hide_fraction).index
            relevant_partners = mapping[mapping.index.get_level_values(molecule).isin(ids)]
            relevant_partners = relevant_partners.sample(frac=partner_hide_fraction).index.get_level_values(partner_molecule)
            partner_mask[sample_name] = False
            partner_mask.loc[relevant_partners, sample_name] = True

        return MaskedDataset(
            dataset=dataset,
            masks={molecule: prot_mask},
            hidden={
                molecule: eval_protein_mask,
                partner_molecule: partner_mask,
            },
        )
    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset
