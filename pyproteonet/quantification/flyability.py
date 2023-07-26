from tqdm.auto import tqdm
import pandas as pd

from ..data.dataset import Dataset

def flyability_upper_bound(dataset: Dataset, column: str = 'abundance', mapping: str = 'protein', remove_one: bool = False):
    mapped = dataset.molecule_set.get_mapped_pairs(molecule_a='protein', molecule_b='peptide', mapping=mapping)
    unique_peps = dataset.molecule_set.get_mapping_unique_molecules(molecule='peptide', partner_molecule='protein', mapping=mapping)
    mapped = mapped[mapped.peptide.isin(unique_peps)] 
    max_divided = []
    for sample in tqdm(dataset.samples):
        mapped[column] = sample.values['peptide'][column].loc[mapped.peptide].values
        groups = mapped.groupby('protein')[column]
        mapped['max'] = groups.max().loc[mapped['protein']].values
        mapped['sum'] = groups.sum().loc[mapped['protein']].values
        mapped['count'] = groups.count().loc[mapped['protein']].values
        max_d = (mapped[column] / mapped['max'])[mapped['count'] > 1]
        if remove_one:
            max_d = max_d[max_d!=1]
        max_divided.append(max_d)
    max_divided = pd.concat(max_divided)
    max_divided.name = 'flyability_upper_bound'
    return max_divided