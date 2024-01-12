from typing import Any, Dict, Tuple, Callable, List, Iterable, Optional, Union
from collections import OrderedDict
import glob
import shutil

import numpy as np
import pandas as pd
from pandas import HDFStore
from pathlib import Path
import json


from .molecule_set import MoleculeSet, MoleculeMapping
from .dataset_sample import DatasetSample
from ..utils.numpy import eq_nan
from ..utils.pandas import matrix_to_multiindex
from ..processing.dataset_transforms import rename_values, drop_values, rename_columns


class DatasetMoleculeValues:
    def __init__(self, dataset: "Dataset", molecule: str):
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
        self.values = {
            molecule: DatasetMoleculeValues(self, molecule)
            for molecule in self.molecules.keys()
        }
        self._dgl_graph = None

    @classmethod
    def load(cls, dir_path: Union[str, Path])->"Dataset":
        """loads a previsously saved dataset from disk

        Args:
            dir_path (Union[str, Path]): path to the directory representing the dataset

        Returns:
            Dataset: the loaded dataset
        """
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

    @classmethod
    def from_mapped_dataframe(
        cls,
        df: pd.DataFrame,
        molecule: str,
        sample_columns: List[str],
        id_column: Optional[str] = None,
        result_column_name: str = "abundance",
        mapping_column: Optional[str] = None,
        mapping_sep: str = ",",
        partner_molecule: str = "protein",
        mapping_name="peptide-protein",
    ) -> "Dataset":
        """Transforming a pandas dataframe into a dataset. Useful for loading tabular peptide abundance data with a mapping column mapping peptides to proteins.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            molecule (str): The molecule whose values are contained in the dataframe.
            sample_columns (List[str]): The list of columns representing the dataset samples
            id_column (Optional[str], optional): The name of the column representing molecule ids. If none the dataframe index is used. Defaults to None.
            result_column_name (str, optional): Name of the value column used for the loaded values. Defaults to "abundance".
            mapping_column (Optional[str], optional): Column containing lists of partner molecule ids. Defaults to None.
            mapping_sep (str, optional): The seperator character used to separed partner molecuel ids. Defaults to ",".
            mapping_molecule (str, optional): The name of the partner molecule type. Defaults to "protein".
            mapping_name (str, optional): The name of the mapping created from the mapping_column. Defaults to "peptide-protein".

        Returns:
            Dataset: the loaded dataset.
        """
        from ..io.io import read_mapped_dataframe
        return read_mapped_dataframe(
            df=df,
            molecule=molecule,
            sample_columns=sample_columns,
            id_column=id_column,
            result_column_name=result_column_name,
            mapping_column=mapping_column,
            mapping_sep=mapping_sep,
            mapping_molecule=partner_molecule,
            mapping_name=mapping_name,
        )

    def save(self, dir_path: Union[str, Path], overwrite: bool = False):
        """Saves the dataset to disk as a directory containing .h5 files for the samples and a .h5 file for the molecule set.

        Args:
            dir_path (Union[str, Path]): Directory path to save the dataset to.
            overwrite (bool, optional): Wheter to overwrite any existing data. Defaults to False.

        Raises:
            FileExistsError: Raised if the directory already exists and overwrite is False.
        """
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
        """Write .tsv files for the given molecules and columns to the given directory.

        Args:
            output_dir (Path): The output directory path.
            molecules (List[str], optional): The molecules whose columns should be written to .tsv files. Defaults to ["protein", "peptide"].
            columns (List[&quot;str&quot;], optional): The column to write. Every column produces a .tsv file the with column values for every samples. Defaults to ["abundance"].
            molecule_columns (Union[bool, List[str]], optional): Any columns from the MoleculeSet to add to the .tsv files. Defaults to [].
            index_names (Optional[List[str]], optional): How to name the index columns in the .tsv files. Defaults to None.
            na_rep (str, optional): How to represent missing (NaN) values in the .tsv files. Defaults to "NA".
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if index_names is None:
            index_names = molecules
        for molecule, index_name in zip(molecules, index_names):
            for column in columns:
                vals = self.get_samples_value_matrix(
                    molecule=molecule, column=column, molecule_columns=molecule_columns
                )
                vals.index.set_names(index_name, inplace=True)
                vals = vals.reset_index()
                vals.to_csv(
                    output_dir / f"{molecule}_{column}.tsv",
                    sep="\t",
                    na_rep=na_rep,
                    index=False,
                )

    def __getitem__(self, sample_name: str) -> DatasetSample:
        return self.samples_dict[sample_name]

    def create_sample(self, name: str, values: Dict[str, pd.DataFrame]):
        """Add a new sample to the dataset.

        Args:
            name (str): The name of the sample to add.
            values (Dict[str, pd.DataFrame]): The values for the sample. The keys are the molecule types and the values are dataframes with the molecule ids as index and the values as columns.

        Raises:
            ValueError: Raised if index of the given dataframes does not align with the molecule ids of the dataset.
        """
        if name in self.samples_dict:
            KeyError(f"Sample with name {name} already exists.")
        for mol, mol_df in self.molecules.items():
            if mol not in values:
                values[mol] = pd.DataFrame(index=mol_df.index)
            else:
                if not values[mol].index.isin(mol_df.index).all():
                    raise ValueError(
                        f"The dataframe for molecule {mol} contains an index which is not in the molecule set's molecule ids for {mol}."
                    )
                values[mol] = pd.DataFrame(data=values[mol], index=mol_df.index)
            values[mol].index.name = "id"
        for key, vals in [
            (key, vals)
            for key, vals in values.items()
            if key not in self.molecules.keys()
        ]:
            values[key] = vals
        self.samples_dict[name] = DatasetSample(dataset=self, values=values, name=name)

    @property
    def samples(self) -> Iterable[DatasetSample]:
        return self.samples_dict.values()

    @property
    def num_samples(self) -> int:
        return len(self.samples_dict)

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
    def mappings(self) -> Dict[str, MoleculeMapping]:
        return self.molecule_set.mappings

    def number_molecules(self, molecule: str) -> int:
        """The number of molecules for a given molecule type.

        Args:
            molecule (str): The molecule type to get the number of molecules for (e.g. protein, peptide ...)

        Returns:
            int: The number of molecules.
        """
        return self.molecule_set.number_molecules(molecule=molecule)

    def __len__(self) -> int:
        return len(self.samples_dict)

    def __iter__(self) -> Iterable:
        return self.samples

    def sample_apply(self, fn: Callable, *args, **kwargs):
        """Apply a function for every dataset samples

        Args:
            fn (Callable): The function to apply.

        Returns:
            _type_: The transformed dataset.
        """
        transformed = {}
        for key, sample in self.samples_dict.items():
            transformed[key] = fn(sample, *args, **kwargs)
        return Dataset(molecule_set=self.molecule_set, samples=transformed)

    def copy(
        self,
        samples: Optional[List[str]] = None,
        columns: Optional[
            Union[Iterable[str], Dict[str, Union[str, Iterable[str]]]]
        ] = None,
        copy_molecule_set: bool = True,
        molecule_ids: Dict[str, pd.Index] = {},
    ):
        """Copies the dataset.

        Args:
            samples (Optional[List[str]], optional): Dataset samples to include in the copy (all samples if not given). Defaults to None.
            columns (Optional[ Union[Iterable[str], Dict[str, Union[str, Iterable[str]]]] ], optional): Which value columns to copy for every molecule. Defaults to None.
            copy_molecule_set (bool, optional): Wheter to copy the MoleculeSet or just store a reference to the original MoleculeSet. Defaults to True.
            molecule_ids (Dict[str, pd.Index], optional): Which molecule ids to copy for every molecule type (all molecule ids are copied if a molecule type is not specified). Defaults to {}.

        Returns:
            _type_: _description_
        """
        copied = {}
        samples_dict = self.samples_dict
        if samples is None:
            samples = self.sample_names
        for name in samples:
            sample = samples_dict[name]
            copied[name] = sample.copy(columns=columns, molecule_ids=molecule_ids)
        molecule_set = self.molecule_set
        if copy_molecule_set:
            molecule_set = molecule_set.copy(molecule_ids=molecule_ids)
        return Dataset(molecule_set=molecule_set, samples=copied)

    def get_molecule_subset(self, molecule: str, ids: pd.Index):
        """Create a new dataset containing only the given molecule ids for the given molecule type.

        Args:
            molecule (str): The molecule type to copy
            ids (pd.Index): The molecule ids to copy

        Returns:
            Dataset: A new dataset containing the specified subset of the old dataset
        """
        return self.copy(molecule_ids={molecule: ids}, copy_molecule_set=True)

    def lf(
        self,
        molecule: str,
        columns: Optional[List[str]] = None,
        molecule_columns: List[str] = [],
    ):
        """Returns a dataframe in long format with multindex (sample id, molecule id) representing the value columns for the specified molecule type.

        Args:
            molecule (str): The molecule type (e.g. protein, peptide ...)
            columns (Optional[List[str]], optional): The value columns to include in the result, default to all vall columns. Defaults to None.
            molecule_columns (List[str], optional): Any molecule columns from the MoleculeSet to include in the resulting dataframe. Defaults to [].

        Returns:
            pd.DataFrame: the resulting dataframe
        """
        return self.get_values_flat(
            molecule=molecule, columns=columns, molecule_columns=molecule_columns
        )

    def wf(self, molecule: str, column: str)->pd.DataFrame:
        """Returns a dataframe in wide format (molecule ids as index, sample names as columns) representing the values of the specified value column for the specified molecule type.

        Args:
            molecule (str): The molecule type (e.g. protein, peptide ...)
            column (Optional[List[str]], optional): The value column to use.

        Returns:
            pd.DataFrame: the resulting dataframe
        """
        return self.get_samples_value_matrix(molecule=molecule, column=column)

    def get_values_flat(
        self,
        molecule: str,
        columns: Optional[List[str]] = None,
        molecule_columns: List[str] = [],
    )->pd.DataFrame:
        """Returns a dataframe in long format with multindex (sample id, molecule id) representing the value columns for the specified molecule type.

        Args:
            molecule (str): The molecule type (e.g. protein, peptide ...)
            columns (Optional[List[str]], optional): The value columns to include in the result, default to all vall columns. Defaults to None.
            molecule_columns (List[str], optional): Any molecule columns from the MoleculeSet to include in the resulting dataframe. Defaults to [].

        Returns:
            pd.DataFrame: the resulting dataframe
        """
        sample_names, df = [], []
        for name, sample in self.samples_dict.items():
            if columns is None:
                values = sample.values[molecule].copy()
            else:
                values = sample.values[molecule][columns].copy()
            if molecule_columns:
                if set.intersection(set(values.columns), set(molecule_columns)):
                    raise AttributeError(
                        "There are columns and molecule columns with identical name"
                    )
                values[molecule_columns] = self.molecules[molecule][molecule_columns]
            sample_names.extend(np.full(len(values), name))
            df.append(values)
        df = pd.concat(df)
        index = pd.DataFrame({"sample": sample_names, "id": df.index})
        index = pd.MultiIndex.from_frame(index)
        df.set_index(index, inplace=True)
        return df

    def infer_mapping(self, molecule: str, mapping: str) -> Tuple[str, str, str]:
        """Infer a mapping name from a molecule type and a mapping string.

        Args:
            molecule (str): Molecule type like protein, peptide ...
            mapping (str): If the name of a molecule type is given it is tried to infer the mapping name connecting both molecule types. If a mapping name is given it is returned as is.

        Returns:
            Tuple[str, str, str]: The from molecule type, the mapping name, and the to molecule type.
        """
        return self.molecule_set.infer_mapping(molecule=molecule, mapping=mapping)

    def get_mapping_partner(self, molecule: str, mapping: str) -> str:
        """Infer the partner molecule type for a molecule type and mapping

        Args:
            molecule (str): The one molecule type of the mapping
            mapping (str): The mapping name.

        Returns:
            str: The other molecule type of the mapping.
        """
        return self.molecule_set.get_mapping_partner(molecule=molecule, mapping=mapping)

    def get_mapped(
        self,
        molecule: str,
        mapping: str,
        columns: Union[str, List[str]] = [],
        samples: Optional[List] = None,
        partner_columns: Union[str, List[str]] = [],
        molecule_columns: Union[str, List[str]] = [],
        molecule_columns_partner: Union[str, List[str]] = [],
        return_partner_index_name: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        """Return a dataframe containing all pairs of molecules connected by the given mapping with the values for the corresponding value columns.

        Args:
            molecule (str): A molecule type like protein, peptide...
            mapping (str): A mapping name.
            columns (Union[str, List[str]], optional): The value columns of the given molecule type to include in the results. Defaults to [].
            samples (Optional[List], optional): The names of the samples to include in the results. Defaults to None.
            partner_columns (Union[str, List[str]], optional): The value columns of the partner molecule type to include. Defaults to [].
            molecule_columns (Union[str, List[str]], optional): Any molecule columns from the MoleculeSet to include for the given molecule. Defaults to [].
            molecule_columns_partner (Union[str, List[str]], optional): Any molecule columns from the MoleculeSet to include for the given partner molecule. Defaults to [].
            return_partner_index_name (bool, optional): Whether to return the name of the partner index. Defaults to False.

        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, str]]: the resulting dataframe and an optinal partner index name.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(partner_columns, str):
            partner_columns = [partner_columns]
        if isinstance(molecule_columns, str):
            molecule_columns = [molecule_columns]
        if isinstance(molecule_columns_partner, str):
            molecule_columns_partner = [molecule_columns_partner]
        # if not columns:
        #    raise AttributeError("The list of columns needs to contain at least one column!")
        mapping = self.molecule_set.get_mapping(
            molecule=molecule,
            mapping_name=mapping,
            molecule_columns=molecule_columns,
            partner_columns=molecule_columns_partner,
        )
        # if partner_molecule is None:
            # partner_molecule = [n for n in mapping.mapping_molecules if n not in {'sample', molecule}]
            # assert len(partner_molecule) == 1
        partner_molecule = mapping.mapping_molecules[1]
        cols = set(mapping.df.columns)
        if samples is None:
            samples = self.sample_names
        if "sample" in mapping.df.index.names:
            raise RuntimeError(
                "ERROR"
            )  # TODO: this should never happen, so we might remove it
            sample_maps = {
                sample: m.set_index([molecule, partner_molecule])
                for sample, m in mapped.groupby("sample")
                if sample in samples
            }
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
            vals = map.copy()
            vals[columns] = (
                self.samples_dict[sample]
                .values[molecule]
                .loc[map.index.get_level_values(0), columns]
                .values
            )
            if partner_columns:
                partner_vals = (
                    self.samples_dict[sample]
                    .values[partner_molecule]
                    .loc[map.index.get_level_values(1), partner_columns]
                )
                for pc in partner_vals:
                    vals[pc] = partner_vals[pc].values
            vals = pd.concat([vals], keys=[sample], names=["sample"])
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
    )->pd.Series:
        """Returns a single value columns as a pandas Series with a MultiIndex with the levels: "id" (molecule id) and "sample"

        Args:
            molecule (str): The molecule type to get the values for.
            column (str, optional): The value column to get the values for. Defaults to "abundance".
            samples (Optional[List[str]], optional): The name of the samples to consider or None to consider all samples. Defaults to None.
            ids (Optional[Iterable], optional): The molecule ids to consider. Defaults to None.
            return_missing_mask (bool, optional): Whether to return a mask of missing values. Defaults to False.
            drop_sample_id (bool, optional): Wheter to trop the sample id from the result's index. Defaults to False.

        Returns:
            pd.Series: The resulting pandas Series.
        """
        vals = self.get_samples_value_matrix(
            molecule=molecule,
            column=column,
            molecule_columns=[],
            samples=samples,
            ids=ids,
        )
        vals = matrix_to_multiindex(vals)
        if drop_sample_id:
            vals.reset_index(level="sample", drop=True, inplace=True)
        if return_missing_mask:
            return vals, eq_nan(vals, self.missing_value)
        else:
            return vals

    def set_column_flat(
        self,
        molecule: str,
        values: Union[pd.Series, int, float],
        column: Optional[str] = None,
        skip_foreign_ids: bool = False,
        fill_missing: bool = False,
    ):
        """Sets values from a Pandas Series which has a MultiIndex with the levels: "id" and "sample"

        Args:
            molecule (str): The molecule type to set the values for.
            values (Union[pd.Series, int, float]): The values to set (must either be a pandas Series with a MultiIndex containing the levels "id" and "sample" or a single value).
            column (Optional[str], optional): If given this column name is used
                otherwise the name of the Series is used as column name. Defaults to None.
        """
        if column is None:
            column = values.name
        if isinstance(values, pd.Series):
            for name, group in values.groupby("sample"):
                group = group.droplevel(level="sample")
                sample_values = self.samples_dict[name].values[molecule]
                if (
                    not skip_foreign_ids
                    and not group.index.isin(sample_values.index).all()
                ):
                    raise KeyError(
                        "Some of the provided values have ids that do not exist for this molecule."
                        " If you want to ignore those set the allow_foreign_ids attribute."
                    )
                if fill_missing:
                    sample_values[column] = self.missing_value
                sample_values[column] = group
        else:
            for sample in self.samples:
                sample.values[molecule][column] = values

    def get_samples_value_matrix(
        self,
        molecule: str,
        column: str = "abundance",
        molecule_columns: Union[bool, List[str]] = [],
        samples: Optional[List[str]] = None,
        ids: Optional[Iterable] = None,
    )->pd.DataFrame:
        """Returns a dataframe in wide format (molecule ids as index, sample names as columns) representing the values of the specified value column for the specified molecule type.

        Args:
            molecule (str): The molecule type (e.g. protein, peptide ...)
            column (Optional[List[str]], optional): The value column to use.
            molecule_columns (Union[bool, List[str]], optional): Any molecule columns from the MoleculeSet to include in the resulting dataframe. Defaults to [].
            samples (Optional[List[str]], optional): The names of the samples to consider for the genrated result. Defaults to None.
            ids (Optional[Iterable], optional): The molecule ids to consider for the generated result. Defaults to None.

        Returns:
            pd.DataFrame: the resulting dataframe
        """
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
            sample_df = self.samples_dict[name].values[molecule]
            if column in sample_df.columns:
                res.loc[:, name] = sample_df.loc[:, column]
            else:
                res.loc[:, name] = self.missing_value
        if molecule_columns:
            res.loc[:, molecule_columns] = self.molecules[molecule].loc[
                :, molecule_columns
            ]
        return res

    def set_samples_value_matrix(
        self, matrix: pd.DataFrame, molecule: str, column: str = "abundance"
    ):
        """Sets a dataframe in wide format (molecule ids as index, sample names as columns) for the values of the given value column for the given molecule type.

        Args:
            molecule (str): The molecule type (e.g. protein, peptide ...)
            column (Optional[List[str]], optional): The name of the value column to store the result in.
            
        Returns:
            pd.DataFrame: the resulting dataframe
        """
        for sample_name, sample in self.samples_dict.items():
            if sample_name in matrix.keys():
                sample.values[molecule][column] = matrix[sample_name]

    def rename_molecule(self, molecule: str, new_name: str):
        """Rename a molecule type.

        Args:
            molecule (str): The current name.
            new_name (str): The new name.

        Raises:
            KeyError: Raised when the new name already exists.
        """
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

    def rename_mapping(self, mapping: str, new_name: str):
        """Rename a mapping.

        Args:
            mapping (str): The old name of the mapping.
            new_name (str): The new name of the mapping.
        """
        self.molecule_set.rename_mapping(mapping=mapping, new_name=new_name)

    def rename_columns(
        self, columns: Dict[str, Dict[str, str]], inplace: bool = False
    ) -> Optional["Dataset"]:
        """Rename one or several value columns.

        Args:
            columns (Dict[str, Dict[str, str]]): A dictionary mapping old to new column names for every molecule type (protein, peptide etc.)
            inplace (bool, optional): Whether to perform the operation inplace or return a copy. Defaults to False.

        Returns:
            Optional[Dataset]: A copy of the dataset with the renamed columns if inplace is False, otherwise None.
        """
        return rename_columns(dataset=self, columns=columns, inplace=inplace)

    def rename_values(
        self,
        columns: Dict[str, str],
        molecules: Optional[List[str]] = None,
        inplace: bool = False,
    ):
        """Similar to rename_columns but uses the same mapping for all molecule types."""
        return rename_values(
            data=self, columns=columns, molecules=molecules, inplace=inplace
        )

    def drop_values(
        self,
        columns: List[str],
        molecules: Optional[List[str]] = None,
        inplace: bool = False,
    )->Optional["Dataset"]:
        """Drop one or several value columns.

        Args:
            columns (List[str]): The columns to drop.
            molecules (Optional[List[str]], optional): The molecules for which the given columns are dropped if they exist. Defaults to None.
            inplace (bool, optional): Whether to return a new dataset. Defaults to False.

        Returns:
            _type_: The resulting dataset if inplace is False, otherwise None.
        """
        return drop_values(
            data=self, columns=columns, molecules=molecules, inplace=inplace
        )

    def to_dgl_graph(
        self,
        feature_columns: Dict[str, Union[str, List[str]]],
        mappings: Union[str, List[str]],
        mapping_directions: Dict[str, Tuple[str, str]] = {},
        make_bidirectional: bool = False,
        features_to_float32: bool = True,
        samples: Optional[List[str]] = None,
    ) -> "dgl.DGLHeteroGraph":
        """Transform the dataset into a dgl graph.

        Args:
            feature_columns (Dict[str, Union[str, List[str]]]): value columns to include as features for the nodes of the graph.
            mappings (Union[str, List[str]]): Names of the mappings to use for the edges of the graph.
            mapping_directions (Dict[str, Tuple[str, str]], optional): Used to specifies the direction of edges between molecule types. Defaults to {}.
            make_bidirectional (bool, optional): Whether to make the graph edges bidirectional. Defaults to False.
            features_to_float32 (bool, optional): Cast all feature values to float32. Defaults to True.
            samples (Optional[List[str]], optional): The names of the samples to include in the graph. If not given all samples are included. Defaults to None.

        Raises:
            KeyError: Raised if feature columns with the reserved names 'hidden' and 'mask' are specified

        Returns:
            dgl.DGLHeteroGraph: the created graph
        """
        import dgl
        import torch
        if samples is None:
            samples = self.sample_names
        graph_data = dict()
        if isinstance(mappings, str):
            mappings = [mappings]
        for mapping_name in mappings:
            mapping = self.mappings[mapping_name]
            if mapping_name in mapping_directions:
                if tuple(mapping_directions[mapping_name]) != mapping.mapping_molecules:
                    mapping = mapping.swaplevel()
            if not make_bidirectional:
                edge_mappings = [mapping]
            else:
                edge_mappings = [mapping, mapping.swaplevel()]
            for mapping in edge_mappings:
                identifier = (
                    mapping.mapping_molecules[0],
                    mapping_name,
                    mapping.mapping_molecules[1],
                )
                edges = []
                for i, mol in enumerate(mapping.mapping_molecules):
                    e_data = self.molecules[mol].index.get_indexer(
                        mapping.df.index.get_level_values(i)
                    )
                    edges.append(torch.from_numpy(e_data))
                edges = tuple(edges)
                graph_data[identifier] = edges
        g = dgl.heterograph(graph_data)
        for mol, mol_features in feature_columns.items():
            if isinstance(mol_features, str):
                mol_features = [mol_features]
            mol_ids = self.molecules[mol].index
            for feature in mol_features:
                if feature in {"hidden", "mask"}:
                    raise KeyError(
                        'Feature names "hidden" and "mask" are reserved names'
                    )
                mat = self.get_samples_value_matrix(molecule=mol, column=feature).loc[
                    mol_ids, samples
                ]
                feat = torch.from_numpy(mat.to_numpy())
                if features_to_float32:
                    feat = feat.to(torch.float32)
                g.nodes[mol].data[feature] = feat
        # if samples is None:
        #     res = g
        # else:
        #     res = copy.deepcopy(g)
        #     sample_ids = {sample: i for i, sample in enumerate(self.sample_names)}
        #     ids = torch.tensor([sample_ids[sample] for sample in samples])
        #     for mol, mol_features in feature_columns.items():
        #         for feature in mol_features:
        #             sample_mat = res.nodes[mol].data[feature][:, ids]
        #             res.nodes[mol].data[feature] = sample_mat
        return g

    def calculate_hist(
        self, molecule_name: str, bins="auto"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate a histogram for the values of a given molecule type.

        Args:
            molecule_name (str): The molecule type to generate the histogram for.
            bins (str, optional): The bins. Defaults to "auto".

        Returns:
            Tuple[np.ndarray, np.ndarray]: The histogram values.
        """
        values = self.values[molecule_name]
        mask = ~values.isna()
        existing = values[mask]
        bin_edges = np.histogram_bin_edges(existing, bins=bins)
        hist = np.histogram(values, bins=bin_edges)
        return hist
