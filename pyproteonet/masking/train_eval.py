from typing import Tuple

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset
from .masked_dataset_generator import MaskedDatasetGenerator
from .missing_values import mask_non_missing


def train_eval_protein_and_mapped(
    dataset: Dataset,
    protein_abundance_column: str,
    peptide_abundance_column: str,
    validation_fraction=0.1,
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

    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(
            molecule=protein_molecule, column=protein_abundance_column
        )
        relevant_peptides = dict()
        for sample in input_dataset.sample_names:
            ids = prot_mask[sample].isna()
            ids = ids[~ids]
            ids = (
                ids[~ids.index.isin(validation_proteins)]
                .sample(frac=training_fraction)
                .index
            )
            prot_mask[sample] = False
            prot_mask.loc[ids, sample] = True
            relevant_peptides[sample] = mapping[
                mapping.index.get_level_values(0).isin(ids)
            ].index.get_level_values(1)
        pep_mask = input_dataset.get_samples_value_matrix(
            molecule=peptide_molecule, column=peptide_abundance_column
        )
        for sample in input_dataset.sample_names:
            ids = pep_mask[sample].isna()
            ids = ids[~ids]
            ids = ids[ids.index.isin(relevant_peptides[sample])]
            ids = ids[~ids.index.isin(validation_peptides)]
            ids = ids.sample(frac=peptide_masking_fraction).index
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
