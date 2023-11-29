import numpy as np

from ..data.dataset import Dataset
from ..processing.molecule_set_transforms import num_theoretical_peptides

def iBAQ(
    dataset: Dataset,
    input_column: str = 'abundance',
    sequence_column: str = "sequence",
    enzyme: str = "Trypsin",
    min_peptide_length: int = 6,
    max_peptide_length: int = 30,
    result_column: str = "abundance",
    only_unique: bool = False,
    mapping: str = 'protein',
    inplace: bool = False,
    tqdm_bar: bool = False
):
    num_peptides = num_theoretical_peptides(molecule_set=dataset.molecule_set, min_peptide_length=min_peptide_length, max_peptide_length=max_peptide_length,
        enzyme=enzyme, sequence_column=sequence_column, result_column=None)
    res = sum(dataset=dataset, only_unique=only_unique, input_column=input_column, result_column=result_column, mapping=mapping, inplace=inplace, tqdm_bar=tqdm_bar)
    if not inplace:
        dataset = res # type: ignore
    for sample in dataset.samples:
        prot_vals = sample.values['protein']
        prot_vals[result_column] /= num_peptides.loc[prot_vals.index]
        mask = prot_vals[result_column] == np.inf
        prot_vals.loc[mask, result_column] = dataset.missing_value
    if not inplace:
        return dataset