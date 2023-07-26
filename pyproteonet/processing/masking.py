from typing import Optional, Tuple, Iterable, Union, List

from numpy.random import Generator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, train_test_split
import pandas as pd
import numpy as np

from ..data.dataset import Dataset
from ..data.masked_dataset import MaskedDataset

def molecule_mask(dataset: Dataset, molecules_to_mask: Union[pd.Index, List, np.ndarray], molecule: str = 'protein'):
    mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    for sample_name, sample in dataset.samples_dict.items():
        mask[sample_name] = False
        mask.loc[molecules_to_mask, sample_name] = True
    return MaskedDataset(dataset=dataset, mask=mask)

def train_test_full(
    dataset: Dataset, train_size: float = 0.8, molecule: str = "protein", random_state: Optional[int] = None
)->Tuple[MaskedDataset, MaskedDataset]:
    splitter = ShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    ms = dataset.molecule_set
    train_molecules, test_molecules = next(splitter.split(ms.molecules[molecule]))
    test_molecules = ms.molecules[molecule].iloc[test_molecules]
    train_molecules = ms.molecules[molecule].iloc[train_molecules]
    train_masks = pd.DataFrame(index=ms.molecules[molecule].index)
    test_masks = pd.DataFrame(index=ms.molecules[molecule].index)
    for sample_name, sample in dataset.samples_dict.items():
        train_masks[sample_name] = False
        train_masks.loc[train_molecules.index, sample_name] = True
        test_masks[sample_name] = False
        test_masks.loc[train_molecules.index, sample_name] = True
    train_ds = MaskedDataset(dataset=dataset, mask=train_masks)
    test_ds = MaskedDataset(dataset=dataset, mask=test_masks, hidden=train_masks)
    return train_ds, test_ds

def train_test_non_missing_no_sample_overlap(
    dataset: Dataset, train_molecules_size: float = 0.8, molecule: str = "protein", column: str = 'abundance', random_state: Optional[int] = None
)->Tuple[MaskedDataset, MaskedDataset]:
    splitter = ShuffleSplit(n_splits=1, train_size=train_molecules_size, random_state=random_state)
    ms = dataset.molecule_set
    train_molecules, test_molecules = next(splitter.split(ms.molecules[molecule]))
    test_molecules = ms.molecules[molecule].iloc[test_molecules]
    train_molecules = ms.molecules[molecule].iloc[train_molecules]
    train_masks = pd.DataFrame(index=ms.molecules[molecule].index)
    test_masks = pd.DataFrame(index=ms.molecules[molecule].index)
    for sample_name, sample in dataset.samples_dict.items():
        missing = sample.missing_molecules(molecule=molecule, column=column)
        train_masks[sample_name] = False
        train_masks.loc[train_molecules.index, sample_name] = True
        train_masks.loc[missing.index, sample_name] = False
        test_masks[sample_name] = False
        test_masks.loc[test_molecules.index, sample_name] = True
        test_masks.loc[missing.index, sample_name] = False
    train_ds = MaskedDataset(dataset=dataset, mask=train_masks, hidden=test_masks)
    test_ds = MaskedDataset(dataset=dataset, mask=test_masks, hidden=train_masks)
    return train_ds, test_ds



# def create_protein_masks_stratified_fold(data_set: 'DatasetSample', shuffle: bool = True,
#                                          num_bins: int = 10, num_folds: int = 10,
#                                          fraction_to_take: float = 1.0):
#     protein_values = data_set.values['protein']
#     protein_values = protein_values[~eq_nan(protein_values.abundance, data_set.missing_abundance_value)]
#     if not shuffle:
#         protein_values = protein_values.iloc[:int(protein_values.shape[0] * fraction_to_take)]
#     else:
#         protein_values = protein_values.sample(frac=fraction_to_take)
#     discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
#     label = discretizer.fit_transform(protein_values.abundance.to_numpy()[:, np.newaxis]).squeeze()
#     skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle)
#     splits = skf.split(protein_values, label)
#     train_node_folds = []
#     test_node_folds = []
#     for train_split, test_split in splits:
#         train_split = protein_values.iloc[train_split].index
#         train_node_folds.append(data_set.molecule_set.node_mapping['protein'].loc[train_split].node_id.to_numpy())
#         test_split = protein_values.iloc[test_split].index
#         test_node_folds.append(data_set.molecule_set.node_mapping['protein'].loc[test_split].node_id.to_numpy())
#     return train_node_folds, test_node_folds

# def create_protein_masks_stratified_fold_legacy(data_set: 'DatasetSample', missing_label = -100.0, 
#                                          validation_fraction: float = 0.1, shuffle: bool = True, num_folds: int = 10):
#     node_values = data_set.get_node_values()
#     protein_nodes = node_values[node_values['type'] == NODE_TYPE_MAPPING['protein']]
#     non_missing_protein_nodes = protein_nodes[protein_nodes['label']!=missing_label]
#     if shuffle:
#         non_missing_protein_nodes = non_missing_protein_nodes.sample(frac=1.0)
#     validation_cut = int(non_missing_protein_nodes.shape[0] * validation_fraction)
#     validation_protein_nodes = non_missing_protein_nodes.iloc[:validation_cut].index.to_numpy()
#     train_test_protein_nodes = non_missing_protein_nodes.iloc[validation_cut:]
#     skf = StratifiedKFold(n_splits=num_folds)
#     splits = skf.split(train_test_protein_nodes, train_test_protein_nodes['label'].astype(int))
#     train_test_protein_nodes = train_test_protein_nodes.index.to_numpy()
#     train_node_folds = []
#     test_node_folds = []
#     for train_split, test_split in splits:
#         train_node_folds.append(train_test_protein_nodes[train_split])
#         test_node_folds.append(train_test_protein_nodes[test_split])
#     return train_node_folds, test_node_folds, validation_protein_nodes
#     #validation_protein_nodes = 
#     #missing_label_nodes = node_values[node_values['label'] == missing_label].index

#     if validation_proportion > 0:
#         if use_stratified:
#             splitter = StratifiedShuffleSplit(n_splits=1, train_size=validation_proportion)
#             val, train_test = next(splitter.split(protein_nodes, data_set.values['protein'].loc[protein_nodes.index, 'label']))
#         else:
#             splitter = ShuffleSplit(n_splits=1, train_size=validation_proportion)
#             val, train_test = next(splitter.split(protein_nodes))
#         val = protein_nodes.iloc[val]['node_id'].to_numpy()
#         train_test = protein_nodes.iloc[train_test]
#     else:
#         val = np.array([])


# def create_protein_masks_randomized_split(data_set, train_proportion: float = 0.9, hold_out_proteins: pd.DataFrame = pd.DataFrame(),
#                                           n_splits: int = 1, use_stratified: bool = False, ignore_missing_values: bool = False,
#                                           missing_value_constant: float = np.nan):
#     if ignore_missing_values:
#         protein_nodes = data_set.values['protein']
#         protein_nodes = protein_nodes[protein_nodes['abundance'] != missing_value_constant]
#         protein_nodes = data_set.node_mapping['protein'].loc[protein_nodes.index]
#     else:
#         protein_nodes = data_set.node_mapping['protein']
#     train_test = protein_nodes[~protein_nodes.index.isin(hold_out_proteins.index)]
#     train_masks = []
#     test_masks = []
#     if use_stratified:
#         splitter = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_proportion)
#         label = data_set.values['protein'].loc[train_test.index, 'label']
#     else:
#         splitter = ShuffleSplit(n_splits=n_splits, train_size=train_proportion)
#         label=None
#     for train, test in splitter.split(train_test, label):
#         train_masks.append(train_test.iloc[train]['node_id'].to_numpy())
#         test_masks.append(train_test.iloc[test]['node_id'].to_numpy())
#     return train_masks, test_masks