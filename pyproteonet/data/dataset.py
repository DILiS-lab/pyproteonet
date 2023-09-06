from typing import Any, Dict, Tuple, Callable, List, Iterable, Optional, Union
from collections import OrderedDict
import glob
import shutil

import numpy as np
import pandas as pd
from pandas import HDFStore
from matplotlib import pyplot as plt
import seaborn as sbn
from scipy.stats import pearsonr  # type: ignore
from pathlib import Path
import json

from .molecule_set import MoleculeSet
from .dataset_sample import DatasetSample
from ..utils.numpy import eq_nan
from ..processing.dataset_transforms import rename_values, drop_values

class DatasetMoleculeValues:
    def __init__(self, dataset: 'Dataset', molecule: str):
        self.dataset = dataset
        self.molecule = molecule

    def __getitem__(self, key):
        return self.dataset.get_column_flat(molecule=self.molecule, column=key)
    
    def __setitem__(self, key, values):
        self.dataset.set_column_flat(molecule=self.molecule, values=values, column=key)

    @property
    def df(self):
        return self.dataset.get_values_flat(molecule=self.molecule)


class Dataset:
    """Representing a dataset consisting of a MoleculeSet specifying molecules and relations
    and several DatasetSamples each holding a set of values for every molecule.
    """

    def __init__(
        self,
        molecule_set: MoleculeSet,
        samples: Dict[str, DatasetSample] = {},
        missing_value: float = np.nan,
    ):
        """Generates a dataset based on a MoleculeSet and an optional list of DatasetSamples.

        Args:
            molecule_set (MoleculeSet): The MoleculeSet this dataset is based on
            samples (Dict[str, DatasetSample], optional): Dictionary of DatasetSamples containing samples for this dataset. Defaults to {}.
            missing_value (float, optional): Value used to represent missing values. Defaults to np.nan.
        """
        self.molecule_set = molecule_set
        self.missing_value = missing_value
        self.missing_label_value = np.nan
        self.samples_dict = OrderedDict(samples)
        for name, sample in self.samples_dict.items():
            sample.dataset = self
            sample.name = name
        self.values = {molecule: DatasetMoleculeValues(self, molecule) for molecule in self.molecules.keys()}

    @classmethod
    def load(cls, dir_path: Union[str, Path]):
        dir_path = Path(dir_path)
        molecule_set = MoleculeSet.load(dir_path / "molecule_set.h5")
        missing_value = np.nan
        with open(dir_path / "dataset_info.json") as f:
            dataset_info = json.load(f)
            missing_value = dataset_info["missing_value"]
        ds = cls(molecule_set=molecule_set, missing_value=missing_value)
        samples = glob.glob(f'{dir_path / "samples"}/*.h5')
        samples.sort()
        for sample in samples:
            sample_path = Path(sample)
            values = {}
            with HDFStore(sample_path) as store:
                for molecule in store.keys():
                    values[molecule.strip("/")] = store[molecule]
            ds.create_sample(name=sample_path.stem, values=values)
        return ds

    def save(self, dir_path: Union[str, Path], overwrite: bool = False):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=overwrite)
        samples_dir = dir_path / "samples"
        if samples_dir.exists():
            if overwrite:
                shutil.rmtree(samples_dir)
            else:
                raise FileExistsError(f"{samples_dir} already exists")
        samples_dir.mkdir()
        self.molecule_set.save(dir_path / "molecule_set.h5", overwrite=overwrite)
        with open(dir_path / "dataset_info.json", "w") as f:
            json.dump({"missing_value": self.missing_value}, f)
        for sample_name, sample in self.samples_dict.items():
            with HDFStore(dir_path / "samples" / f"{sample_name}.h5") as store:
                for molecule, df in sample.values.items():
                    store[f"{molecule}"] = df

    def write_tsvs(
        self,
        output_dir: Path,
        molecules: List[str] = ["protein", "peptide"],
        columns: List["str"] = ["abundance"],
        molecule_columns: Union[bool, List[str]] = [],
        index_names: Optional[List[str]] = None,
        na_rep="NA",
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        if index_names is None:
            index_names = molecules
        for molecule, index_name in zip(molecules, index_names):
            for column in columns:
                vals = self.get_samples_value_matrix(molecule=molecule, column=column, molecule_columns=molecule_columns)
                vals.index.set_names(index_name, inplace=True)
                vals = vals.reset_index()
                vals.to_csv(output_dir / f"{molecule}_{column}.tsv", sep="\t", na_rep=na_rep, index=False)

    def __getitem__(self, sample_name: str) -> DatasetSample:
        return self.samples_dict[sample_name]

    def create_sample(self, name: str, values: Dict[str, pd.DataFrame]):
        if name in self.samples_dict:
            KeyError(f"Sample with name {name} already exists.")
        for mol, mol_df in self.molecules.items():
            if mol not in values:
                values[mol] = pd.DataFrame(index=mol_df.index)
            else:
                values[mol] = pd.DataFrame(data=values[mol], index=mol_df.index)
            values[mol].index.name = "id"
        for key, vals in [(key, vals) for key, vals in values.items() if key not in self.molecules.keys()]:
            values[key] = vals
        self.samples_dict[name] = DatasetSample(dataset=self, values=values, name=name)

    @property
    def samples(self) -> Iterable[DatasetSample]:
        return self.samples_dict.values()

    @property
    def sample_names(self) -> List[str]:
        return self.names

    @property
    def names(self) -> List[str]:
        return list(self.samples_dict.keys())

    @property
    def molecules(self) -> Dict[str, pd.DataFrame]:
        return self.molecule_set.molecules

    @property
    def mappings(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        return self.molecule_set.mappings

    def number_molecules(self, molecule: str) -> int:
        return self.molecule_set.number_molecules(molecule=molecule)

    def __len__(self) -> int:
        return len(self.samples_dict)

    def __iter__(self) -> Iterable:
        return self.samples

    def sample_apply(self, fn: Callable, *args, **kwargs):
        transformed = {}
        for key, sample in self.samples_dict.items():
            transformed[key] = fn(sample, *args, **kwargs)
        return Dataset(molecule_set=self.molecule_set, samples=transformed)

    def copy(
        self, samples: Optional[List[str]] = None, columns: Optional[List[str]] = None, copy_molecule_set: bool = True
    ):
        copied = {}
        samples_dict = self.samples_dict
        if samples is None:
            samples = self.sample_names
        for name in samples:
            sample = samples_dict[name]
            copied[name] = sample.copy(columns=columns)
        molecule_set = self.molecule_set
        if copy_molecule_set:
            molecule_set = molecule_set.copy()
        return Dataset(molecule_set=molecule_set, samples=copied)

    def all_values(self, molecule: str, column: str = "abundance", return_missing_mask: bool = False):
        values = []
        mask = []
        for sample in self.samples_dict.values():
            v = sample.values[molecule][column]
            values.append(v)
            if return_missing_mask:
                mask.append(eq_nan(v, sample.missing_value))
        values = pd.concat(values)
        if return_missing_mask:
            return values, np.concatenate(mask)
        return values

    def get_values_flat(self, molecule: str, columns: Optional[List[str]] = None):
        sample_names, df = [], []
        for name, sample in self.samples_dict.items():
            if columns is None:
                values = sample.values[molecule]
            else:
                values = sample.values[molecule][columns]
            sample_names.extend(np.full(len(values), name))
            df.append(values)
        df = pd.concat(df)
        index = pd.DataFrame({"sample": sample_names, "id": df.index})
        index = pd.MultiIndex.from_frame(index)
        df.set_index(index, inplace=True)
        return df
    
    def get_mapping_partner(self, molecule: str, mapping: str)->str:
        return self.molecule_set.get_mapping_partner(molecule=molecule, mapping=mapping)
    
    def get_mapped(self, molecule: str, mapping: str, partner_molecule: str = None, columns: List[str] = [],
                   samples: Optional[List] = None, partner_columns: List[str] = [],
                   molecule_columns:  List[str] = [], molecule_columns_partner: List[str] = [],
                   return_partner_index_name: bool = False)->Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        #if not columns:
        #    raise AttributeError("The list of columns needs to contain at least one column!")
        mapping = self.molecule_set.get_mapping(molecule=molecule, partner_molecule=partner_molecule, mapping_name=mapping,
                                                molecule_columns=molecule_columns, partner_columns=molecule_columns_partner)
        if partner_molecule is None:
            #partner_molecule = [n for n in mapping.mapping_molecules if n not in {'sample', molecule}]
            #assert len(partner_molecule) == 1
            partner_molecule = mapping.mapping_molecules[1]
        cols = set(mapping.df.columns)
        if samples is None:
            samples = self.sample_names
        if 'sample' in mapping.df.index.names:
            raise RuntimeError("ERROR") #TODO: this should never happen, so we might remove it
            sample_maps = {sample: m.set_index([molecule, partner_molecule]) for sample,m in mapped.groupby('sample') if sample in samples}
        else:
            sample_maps = {}
            for sample in samples:
                sample_maps[sample] = mapping.df.copy()
        if cols.intersection(columns):
            raise AttributeError("Result would have duplicated column names!")
        cols.update(columns)
        if cols.intersection(partner_columns):
            raise AttributeError("Result would have duplicated column names!")
        res = []
        for sample, map in sample_maps.items():
            vals = self.samples_dict[sample].values[molecule].loc[map.index.get_level_values(0), columns]
            vals.set_index(map.index, inplace=True)
            if partner_columns:
                partner_vals = self.samples_dict[sample].values[partner_molecule].loc[map.index.get_level_values(1), partner_columns]
                for pc in partner_vals:
                    vals[pc] = partner_vals[pc].values
            vals = pd.concat([vals], keys=[sample], names=['sample'])
            res.append(vals)
        res = pd.concat(res)
        if return_partner_index_name:
            return res, partner_molecule
        else:
            return res

    def get_column_flat(
        self,
        molecule: str,
        column: str = "abundance",
        samples: Optional[List[str]] = None,
        ids: Optional[Iterable] = None,
        return_missing_mask: bool = False,
        drop_sample_id: bool = False,
    ):
        vals = self.get_samples_value_matrix(molecule=molecule, column=column, molecule_columns=[],
                                             samples=samples, ids=ids).stack(dropna=False)
        vals.index.set_names(["id", "sample"], inplace=True)
        vals = vals.swaplevel()
        if drop_sample_id:
            vals.reset_index(level='sample', drop=True, inplace=True)
        if return_missing_mask:
            return vals, eq_nan(vals, self.missing_value)
        else:
            return vals

    def missing_mask(self, molecule: str, column: str = "abundance"):
        return eq_nan(self.get_column_flat(molecule=molecule, column=column), self.missing_value)

    def non_missing_mask(self, molecule: str, column: str = "abundance"):
        return ~self.missing_mask(molecule=molecule, column=column)

    def set_column_flat(self, molecule: str, values: pd.Series, column: Optional[str] = None):
        """Sets values from a Pandas Series which has a MultiIndex with the levels: "id" and "sample"

        Args:
            molecule (str): The molecule type to set the values for.
            values (pd.Series): The values to set (must be a pandas Series with a MultiIndex containing the levels "id" and "sample").
            column (Optional[str], optional): If given this column name is used
                otherwise the name of the Series is used as column name. Defaults to None.
        """
        if column is None:
            column = values.name
        for name, group in values.groupby("sample"):
            group = group.droplevel(level="sample")
            sample_values = self.samples_dict[name].values[molecule]
            #import pdb; pdb.set_trace()
            assert group.index.isin(sample_values.index).all()
            sample_values[column] = group

    def get_samples_value_matrix(self, molecule: str, column: str = "abundance", molecule_columns: Union[bool, List[str]] = [],
                                 samples: Optional[List[str]] = None, ids: Optional[Iterable] = None):
        if samples is None:
            samples = self.sample_names
        if molecule_columns:
            molecule_columns = list(self.molecules[molecule].columns)
        if ids is not None:
            res = self.molecules[molecule].loc[ids, []].copy()
        else:
            res = self.molecules[molecule].loc[:, []].copy()
        for name in samples:
            res[name] = self.missing_value
            res.loc[:, name] = self.samples_dict[name].values[molecule].loc[:, column]
        if molecule_columns:
            res.loc[:, molecule_columns] = self.molecules[molecule].loc[:, molecule_columns]
        return res

    def set_samples_value_matrix(self, matrix: pd.DataFrame, molecule: str, column: str = "abundance"):
        for name, sample in self.samples_dict.items():
            if name in matrix.keys():
                sample.values[molecule][column] = matrix[name]

    def rename_molecule(self, molecule: str, new_name: str):
        if new_name in self.values:
            raise KeyError(f"{new_name} already exists in values.")
        molecule_values = self.values[molecule]
        molecule_values.molecule = new_name
        self.values[new_name] = molecule_values
        del self.values[molecule]
        for sample in self.samples:
            sample.values[new_name] = sample.values[molecule]
            del sample.values[molecule]
        self.molecule_set.rename_molecule(molecule=molecule, new_name=new_name)

    def rename_columns(self, columns: Dict[str, str], molecules: Optional[List[str]] = None, inplace: bool = False):
        return rename_values(data=self, columns=columns, molecules=molecules, inplace=inplace)

    def drop_values(self, columns: List[str], molecules: Optional[List[str]] = None, inplace: bool = False):
        return drop_values(data=self, columns=columns, molecules=molecules, inplace=inplace)

    def create_graph(self, mapping: str = "gene", bidirectional: bool = True, cache: bool = True):
        return self.molecule_set.create_graph(mapping=mapping, bidirectional=bidirectional, cache=cache)

    def calculate_hist(self, molecule_name: str, bins="auto") -> Tuple[np.ndarray, np.ndarray]:
        values, mask = self.all_values(molecule=molecule_name, return_missing_mask=True)
        existing = values[~mask]
        bin_edges = np.histogram_bin_edges(existing, bins=bins)
        hist = np.histogram(values, bins=bin_edges)
        return hist

    def plot_correlation(
        self, molecule: str, column_x: str, column_y: str, samples: Optional[List[str]] = None, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots()
        if samples is None:
            samples = self.sample_names

        plot_df = []
        missing = 0
        for sample in samples:
            sample = self[sample]
            mask_gt = ~sample.missing_mask(molecule, column_x)
            mask_prediction = ~sample.missing_mask(molecule, column_y)
            mask = mask_gt & mask_prediction
            missing += mask_prediction.sum() / self.molecules[molecule].shape[0]
            plot_df.append(sample.values[molecule].loc[mask, [column_x, column_y]])
        missing = missing / len(samples)
        plot_df = pd.concat(plot_df, ignore_index=True)
        sbn.regplot(x=column_x, y=column_y, data=plot_df, ax=ax)
        r, p = pearsonr(plot_df[column_x], plot_df[column_y])
        ax.set_title(f"R2:{round(r**2, 5)}, avg. coverage: {round(missing, 5)   }")
