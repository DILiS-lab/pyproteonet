from typing import Optional, Tuple, Iterable, Union, List, Dict
import collections.abc

from numpy.random import Generator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit, train_test_split
import pandas as pd
import numpy as np

from ..data.dataset import Dataset
from ..data.dataset_sample import DatasetSample
from ..data.masked_dataset import MaskedDataset
from ..data.masked_dataset_iterable import MaskedDatasetIterable
from ..data.masks import DatasetSampleMask
from ..simulation.utils import get_numpy_random_generator


def mask_molecule(
    dataset: Dataset, molecule: str, ids: Optional[Union[pd.Index, List, np.ndarray]] = None
) -> MaskedDataset:
    mds = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    if ids is None:
        ids = dataset.molecule_set.molecules[molecule].index
    for sample_name, sample in dataset.samples_dict.items():
        mds[sample_name] = False
        mds.loc[ids, sample_name] = True
    return MaskedDataset(dataset=dataset, masks={molecule: mds})


def mask_molecule_iterable(
    dataset: Dataset,
    molecule: str,
    frac: Optional[float] = None,
    ids: Optional[Union[pd.Index, List, np.ndarray]] = None,
    hidden_ids: Optional[Union[pd.Index, List, np.ndarray]] = None,
    non_missing_column: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> MaskedDatasetIterable:
    if ids is not None:
        molecules = ids
    else:
        molecules = dataset.molecules[molecule].index
    if hidden_ids is not None:
        hidden_ids = {molecule: hidden_ids}

    def sample(sample: DatasetSample):
        vals = sample.values[molecule]
        vals = vals[vals.index.isin(molecules)]
        if non_missing_column is not None:
            vals = vals.loc[~vals[non_missing_column].isna()]
        if frac is not None:
            sampled_molecules = vals.sample(frac=frac, random_state=random_seed).index
        else:
            sampled_molecules = vals.index
        return DatasetSampleMask(sample=sample, masked={molecule: sampled_molecules}, hidden=hidden_ids)

    MaskedDatasetIterable(dataset=dataset, mask_function=sample, molecule=molecule, has_hidden=hidden_ids is not None)


def mask_missing(
    dataset: Dataset,
    molecule: str = "protein",
    column: str = "abundance",
) -> MaskedDataset:
    mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    for sample_name, sample in dataset.samples_dict.items():
        mask[sample_name] = False
        mask.loc[sample.missing_mask(molecule=molecule, column=column), sample_name] = True
    return MaskedDataset(dataset=dataset, masks={molecule: mask})


def mask_non_missing(
    dataset: Dataset,
    molecule: str,
    column: str,
    frac: float,
    random_seed: Optional[Union[int, np.random.Generator]] = None,
) -> MaskedDataset:
    mask = pd.DataFrame(index=dataset.molecule_set.molecules[molecule].index)
    rng = get_numpy_random_generator(random_seed=random_seed)
    for sample_name, sample in dataset.samples_dict.items():
        mask[sample_name] = False
        masked_mols = sample.non_missing_molecules(molecule=molecule, column=column)
        masked_mols = masked_mols.sample(frac=frac, random_state=rng.integers(np.iinfo(np.int64).max)).index
        mask.loc[masked_mols, sample_name] = True
    return MaskedDataset(dataset=dataset, masks={molecule: mask})


def non_missing_iterable(
    dataset: Dataset,
    molecule: Union[str, Iterable],
    column: str,
    frac: float,
    hidden_ids: Optional[Dict[str, pd.Index]],
) -> MaskedDatasetIterable:
    if isinstance(molecule, str):
        molecule = [molecule]
    hidden_molecule_ids = dict()
    if hidden_ids is not None:
        for mol, ids in hidden_ids.items():
            if "sample" in ids.names:
                hidden_molecule_ids[mol] =  ids.droplevel("sample").groupby(ids.get_level_values("sample"))
            else:
                hidden_molecule_ids[mol] = {sample: ids for sample in dataset.sample_names}

    def sample_train(sample: DatasetSample):
        train_molecules = dict()
        hidden = dict()
        for mol in molecule:
            vals: pd.Series = sample.values[mol].loc[:, column]
            vals = vals[~vals.isna()]
            if mol in hidden_molecule_ids and sample.name in hidden_molecule_ids[mol]:
                hidden[mol] = hidden_molecule_ids[mol][sample.name]
                vals = vals[~vals.index.isin(hidden)]
            train_molecules[mol] = vals.sample(frac=frac).index
        return DatasetSampleMask(sample=sample, masked=train_molecules, hidden=hidden)

    train_ds = MaskedDatasetIterable(dataset=dataset, mask_function=sample_train, has_hidden=True)
    return train_ds


def train_test_non_missing_no_overlap_iterable(
    dataset: Dataset,
    train_frac: float = 0.1,
    test_frac: float = 0.1,
    molecule: Union[str, Iterable] = "protein",
    non_missing_column: str = "abundance",
    ids: Optional[Dict[str, Union[pd.Index, List, np.ndarray]]] = None,
    random_seed: Optional[int] = None,
) -> Tuple[MaskedDatasetIterable, MaskedDataset]:
    """Generates an iterable dataset with random, non-missing train molecules and a fixed set of test molecules.

    A fixed set of test molecules is sampled once for the dataset. For every sample the subset of those test molecues
    that has non-missing values is used as test set. From the remaining non-missing molecules within the sample a
    fraction of size train_frac is samples as train set. Returned datasets are iterable datasets (training molecules
    are not fixed by drawn randomly every time a training sample is requested).

    Args:
        dataset (Dataset): _description_
        train_frac (float, optional): Train set size relative to number of non-missing values of non-test molecules within a sample. Defaults to 0.1.
        test_frac (float, optional): Test set size relative to number of molecules. Defaults to 0.1.
        molecule (str, optional): Molecule type to use. Defaults to "protein".
        non_missing_column (str, optional): Value column to use. Defaults to "abundance".
        ids (Union[pd.Index, List, np.ndarray], optional): If given restricts train and test set to those molecule ids. Defaults to None.
        random_seed (Optional[int], optional): Random state for random draws of train/test sets. Defaults to None.

    Returns:
        Tuple[MaskedDatasetIterable, MaskedDatasetIterable]: Tuple of train and test iterable datasets.
    """
    if isinstance(molecule, str):
        if ids is not None and not isinstance(ids, dict):
            ids = {molecule: ids}
        molecule = [molecule]
    test_molecules = dict()
    overall_train_molecules = dict()
    for mol in molecule:
        available_molecules = dataset.molecules[mol]
        if ids is not None:
            available_molecules = available_molecules.loc[ids[mol]]
        test_molecules[mol] = available_molecules.sample(frac=test_frac, random_state=random_seed).index
        overall_train_molecules[mol] = available_molecules[~available_molecules.index.isin(test_molecules)].index

    def sample_train(sample: DatasetSample):
        train_molecules = dict()
        for mol in molecule:
            vals: pd.Series = sample.values[mol].loc[overall_train_molecules[mol], non_missing_column]
            vals = vals[~vals.isna()]
            train_molecules[mol] = vals.sample(frac=train_frac).index  # TODO: give random seed
        return DatasetSampleMask(sample=sample, masked=train_molecules, hidden=test_molecules)

    def sample_test(sample: DatasetSample):
        test_non_missing = dict()
        for mol in molecule:
            vals: pd.Series = sample.values[mol][non_missing_column][
                sample.non_missing_mask(molecule=mol, column=non_missing_column)
            ]
            test_non_missing[mol] = vals[vals.index.isin(test_molecules[mol])].index
        return DatasetSampleMask(sample=sample, masked=test_non_missing)

    train_ds = MaskedDatasetIterable(dataset=dataset, mask_function=sample_train, has_hidden=True)
    test_ds = MaskedDatasetIterable(dataset=dataset, mask_function=sample_test, has_hidden=False)
    return train_ds, test_ds


def train_test_non_missing_no_sample_overlap(
    dataset: Dataset,
    train_molecules_size: float = 0.8,
    molecule: str = "protein",
    column: str = "abundance",
    random_state: Optional[int] = None,
) -> Tuple[MaskedDataset, MaskedDataset]:
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
    train_ds = MaskedDataset(dataset=dataset, masks={molecule: train_masks}, hidden={molecule: test_masks})
    test_ds = MaskedDataset(dataset=dataset, masks={molecule: test_masks}, hidden={molecule: train_masks})
    return train_ds, test_ds


def train_test_non_missing_no_sample_overlap(
    dataset: Dataset,
    train_molecules_size: float = 0.8,
    molecule: str = "protein",
    column: str = "abundance",
    random_state: Optional[int] = None,
) -> Tuple[MaskedDataset, MaskedDataset]:
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
    train_ds = MaskedDataset(dataset=dataset, masks={molecule: train_masks}, hidden={molecule: test_masks})
    test_ds = MaskedDataset(dataset=dataset, masks={molecule: test_masks}, hidden={molecule: train_masks})
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
