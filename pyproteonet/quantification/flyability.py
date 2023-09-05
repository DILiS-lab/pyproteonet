from tqdm.auto import tqdm
import pandas as pd

from ..data.dataset import Dataset


def estimate_flyability_upper_bound(
    dataset: Dataset,
    column: str = "abundance",
    protein_molecule: str = "protein",
    peptide_molecule: str = "peptide",
    mapping: str = "protein",
    remove_one: bool = True,
    pbar: bool = False,
):
    """Gives an upper bound estimate for the peptide flyability.

    This assumes protein molecules, each consisting of several peptide molecules.
    To get an estimate for peptide flyability unique peptides (peptides that only
    occur within a single protein) are examined. Unique peptides of the same protein
    should all have the same abundance (at least in theory). In reality, they show
    different abundances. We estimate the flyabilies of every peptide by computing 
    its abundance relative to the most abundant unique peptide for the protein.

    Args:
        dataset (Dataset): Dataset to estimate flyability distribution for.
        column (str, optional): Value column to use for the estimation (should represent peptide abundance). Defaults to 'abundance'.
        protein_molecule (str, optional): Name of the molecule in the Dataset that represents proteins. Defaults to 'protein'.
        peptide_molecule (str, optional): Name of the molecule in the Dataset that represents peptides. Defaults to 'peptide'.
        mapping (str, optional): Mapping to use (should represent peptide-to-protein mapping). Defaults to 'protein'.
        remove_one (bool, optional): Per protein all abundances are calculated relative to the highest abundant peptide.
        This results one peptide having a relative abundance of one for every protein (a peak at one in the resulting histogram).
        It makes sense to remove this peak at one from the histogram. Defaults to True.
        pbar (bool, optional): Show progress bar. Defaults to False.

    Returns:
        _type_: The flyability estimate for all peptides that could be estimated.
    """
    mapped = dataset.molecule_set.get_mapped_pairs(
        molecule_a=protein_molecule, molecule_b=peptide_molecule, mapping=mapping
    )
    unique_peps = dataset.molecule_set.get_mapping_unique_molecules(
        molecule=peptide_molecule, partner_molecule=protein_molecule, mapping=mapping
    )
    mapped = mapped[mapped.peptide.isin(unique_peps)]
    max_divided = []
    iterator = dataset.samples
    if pbar:
        iterator = tqdm(iterator)
    for sample in iterator:
        mapped.loc[:, column] = sample.values[peptide_molecule][column].loc[mapped[peptide_molecule]].values
        groups = mapped.groupby(protein_molecule)[column]
        mapped.loc[:, "max"] = groups.max().loc[mapped[protein_molecule]].values
        mapped.loc[:, "sum"] = groups.sum().loc[mapped[protein_molecule]].values
        mapped.loc[:, "count"] = groups.count().loc[mapped[protein_molecule]].values
        max_d = (mapped[column] / mapped["max"])[mapped["count"] > 1]
        if remove_one:
            max_d = max_d[max_d != 1]
        max_divided.append(max_d)
    max_divided = pd.concat(max_divided)
    max_divided.name = "flyability_upper_bound"
    return max_divided
