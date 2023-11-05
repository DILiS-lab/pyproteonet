from typing import Tuple

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset
from .masked_dataset_generator import MaskedDatasetGenerator
from .missing_values import mask_non_missing

def train_proteins_hide_peptides(
    dataset: Dataset,
    protein_abundance_column: str,
    peptide_abundance_column: str,
    validation_fraction=0.1,
    training_fraction=0.1,
    peptide_masking_fraction=0.5,
    protein_molecule="protein",
    peptide_molecule="peptide",
)->Tuple[MaskedDatasetGenerator, MaskedDataset]:
    validation_proteins = dataset.molecules[protein_molecule].sample(frac=validation_fraction).index
    eval_dataset = mask_non_missing(dataset=dataset, molecule=protein_molecule, column=protein_abundance_column, ids=validation_proteins, frac=None)

    def mask(input_dataset: Dataset):
        prot_mask = input_dataset.get_samples_value_matrix(molecule=protein_molecule, column=protein_abundance_column)
        for sample in input_dataset.sample_names:
            ids = prot_mask[sample].isna()
            ids = ids[~ids]
            ids = ids[~ids.index.isin(validation_proteins)].sample(frac=training_fraction).index
            prot_mask[sample] = False
            prot_mask.loc[ids, sample] = True
        pep_mask = input_dataset.get_samples_value_matrix(molecule=peptide_molecule, column=peptide_abundance_column)
        for sample in input_dataset.sample_names:
            ids = pep_mask[sample].isna()
            ids = ids[~ids].sample(frac=peptide_masking_fraction).index
            pep_mask[sample] = False
            pep_mask.loc[ids, sample] = True
        return MaskedDataset(dataset=dataset, masks={protein_molecule: prot_mask, peptide_molecule: pep_mask})
    return MaskedDatasetGenerator(datasets=[dataset], generator_fn=mask), eval_dataset
