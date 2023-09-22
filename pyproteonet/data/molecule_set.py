from typing import Dict, Optional, List, Iterable, Union, Tuple
import uuid
from pathlib import Path
import warnings
from itertools import chain
from dataclasses import dataclass
import shutil

import pandas as pd
from pandas import HDFStore

from .graph_creation import create_graph_nodes_edges
from .molecule_graph import MoleculeGraph


class MoleculeMapping:
    def __init__(self, name: str, df: pd.DataFrame, mapping_molecules: Optional[Tuple[str, str]] = None):
        self.name = name
        self.df = df
        if mapping_molecules is None:
            mapping_molecules = tuple(df.index.names)
        else:
            if mapping_molecules[0] != mapping_molecules[1]:
                assert mapping_molecules[0] == df.index.names[0] and mapping_molecules[1] == df.index.names[1]
            else:
                assert (
                    mapping_molecules[0] + "_a" == df.index.names[0]
                    and mapping_molecules[1] + "_b" == df.index.names[1]
                )
        self.mapping_molecules = mapping_molecules

    def copy(self, molecule_ids: Dict[str, pd.Index] = {}):
        df = self.df.copy()
        if self.mapping_molecules[0] in molecule_ids:
            df = df[df.index.get_level_values(0).isin(molecule_ids[self.mapping_molecules[0]])]
        if self.mapping_molecules[1] in molecule_ids:
            df = df[df.index.get_level_values(1).isin(molecule_ids[self.mapping_molecules[1]])]
        return MoleculeMapping(name=self.name, df=df, mapping_molecules=self.mapping_molecules)

    def add_pairs(self, pairs: pd.DataFrame):
        pairs = pairs.copy()
        pairs.index.set_names(self.df.index.names, inplace=True)
        self.df = pd.concat([self.df, pairs])

    def rename_molecule(self, molecule: str, new_name: str):
        self.mapping_molecules = tuple([(new_name if m == molecule else m) for m in self.mapping_molecules])
        if self.mapping_molecules[0] == self.mapping_molecules[1]:
            self.df.index.set_names((self.mapping_molecules[0] + "_a", self.mapping_molecules[1] + "_b"), inplace=True)
        else:
            self.df.index.set_names(self.mapping_molecules, inplace=True)

    def swaplevel(self):
        mapping_molecules = self.mapping_molecules[::-1]
        df = self.df.swaplevel()
        return MoleculeMapping(name=self.name, df=df, mapping_molecules=mapping_molecules)

    def validate_for_molecule_set(self, molecule_set: "MoleculeSet"):
        assert len(self.df.index.names) == 2
        assert all([m in molecule_set.molecules for m in self.mapping_molecules])
        if self.name in molecule_set.molecules:
            warnings.warn(
                f"Mapping '{self.name}' has the same name as a molecule. This can lead to ambiguity and should be avoided!"
            )
        assert self.df.index.get_level_values(0).isin(molecule_set.molecules[self.mapping_molecules[0]].index).all()
        assert self.df.index.get_level_values(1).isin(molecule_set.molecules[self.mapping_molecules[1]].index).all()
        assert self.df.index.is_unique


def _check_name(name: str):
    if "/" in name:
        raise KeyError('Names must not include "/"!')


class MoleculeSet:
    """A set of molecules and their relations/mapping.
    E.g. a set of proteins and peptides with every peptide belonging to one or many proteins
    """

    def __init__(self, molecules: Dict[str, pd.DataFrame], mappings: Dict[str, Union[pd.DataFrame, MoleculeMapping]]):
        """A set of molecules and their relations/mapping.

        Args:
            molecules (Dict[str, pd.DataFrame]): Relevant molecules as dictionay of molecule name and dataframe containing at least the molecule indices
            mappings (Dict[str, pd.DataFrame]): Every mapping has a multi-index, where every item consists of the ids of the two molecues that are mapped
        """
        self.molecules = molecules
        self.mappings: Dict[str, MoleculeMapping] = {}
        self.mappings_lookup = {}
        for mapping_name, mapping in mappings.items():
            if not isinstance(mapping, MoleculeMapping):
                mapping = MoleculeMapping(name=mapping_name, df=mapping)
            mapping.validate_for_molecule_set(self)
            self.mappings[mapping_name] = mapping
        self._update_mapping_lookup()
        self.clear_cache()
        # self._node_mapping: Optional[Dict[str, pd.DataFrame]] = None
        # self._nodes = None
        # self._edges = None
        self.id = str(uuid.uuid4())

    # def create_graph_nodes_edges(self, mapping:str = 'gene', bidirectional: bool = True):
    #     if self._nodes is None or self._edges is None or self._node_mapping is None:
    #         self._nodes, self._edges, self._node_mapping = create_graph_nodes_edges(molecule_set=self, mapping=mapping, make_bidirectional=bidirectional)
    #     return self._nodes, self._edges, self._node_mapping

    def clear_cache(self):
        self.graphs = dict()

    @property
    def molecule_names(self) -> Iterable[str]:
        return list(self.molecules.keys())

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path)
        molecules = {}
        mappings = {}
        with HDFStore(path) as store:
            keys = store.keys()
            molecule_keys = [key for key in keys if key.split("/")[1] == "molecule"]
            mapping_keys = [key for key in keys if key.split("/")[1] == "mapping"]
            for key in molecule_keys:
                molecules[key.split("/")[-1]] = store[key]
            for key in mapping_keys:
                if len(key.split("/")) == 5:
                    _, _, mapping_name, m1, m2 = key.split("/")
                    mappings[mapping_name] = MoleculeMapping(
                        name=mapping_name, df=store[key], mapping_molecules=(m1, m2)
                    )
                elif len(key.split("/")) == 3:  # legacy format
                    mapping_name = key.split("/")[-1]
                    mapping = store[key]
                    mappings[mapping_name] = MoleculeMapping(
                        name=mapping_name, df=store[key], mapping_molecules=tuple(mapping.index.names)
                    )
                else:
                    raise RuntimeError("MoleculeSet data format not understood.")
        return cls(molecules=molecules, mappings=mappings)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        if path.exists():
            if overwrite:
                path.unlink()
            else:
                raise FileExistsError(f"{path} already exists!")
        with HDFStore(path) as store:
            for molecule, df in self.molecules.items():
                _check_name(molecule)
                store[f"molecule/{molecule}"] = df
            for mapping_name, mapping in self.mappings.items():
                _check_name(mapping_name)
                m1, m2 = mapping.mapping_molecules[0], mapping.mapping_molecules[1]
                _check_name(m1)
                _check_name(m2)
                identifier = f"mapping/{mapping_name}/{m1}/{m2}"
                store[identifier] = mapping.df

    def copy(self, molecule_ids: Dict[str, pd.Index] = {}) -> "MoleculeSet":
        molecules = {}
        for n, v in self.molecules.items():
            if n in molecule_ids:
                v = v[v.index.isin(molecule_ids[n])]
            molecules[n] = v.copy()
        mappings = {}
        for mapping_name, mapping in self.mappings.items():
            mappings[mapping_name] = mapping.copy(molecule_ids=molecule_ids)
        return MoleculeSet(molecules=molecules, mappings=mappings)

    def number_molecules(self, molecule: str) -> int:
        return len(self.molecules[molecule])

    def get_mapped_pairs(self, mapping: str, molecule_a: str = None, molecule_b: str = None) -> pd.DataFrame:
        mapping = self.mappings[mapping]
        if molecule_a is not None and molecule_b is not None:
            assert molecule_a in mapping.mapping_molecules and molecule_b in mapping.mapping_molecules
        mapping = mapping.df
        if len(mapping.columns):
            mapping = mapping.loc[:, []]
        mapping = mapping.reset_index(drop=False)  # TODO: don't do this once everything is refactored
        return mapping

    def infer_mapping(self, molecule: str, mapping: str) -> Tuple[str, str, str]:
        mapping = self.infer_mapping_name(molecule=molecule, mapping_name=mapping)
        partner = [n for n in self.mappings[mapping].mapping_molecules if n != molecule][0]
        return molecule, mapping, partner

    def infer_mapping_name(self, molecule: str, mapping_name: str) -> str:
        if mapping_name not in self.mappings and molecule is not None:
            if molecule not in self.mappings_lookup:
                raise KeyError(f"{molecule} is not a known molecule type, no mapping can be inferred")
            mapping_name = self.mappings_lookup[molecule][mapping_name]
            if len(mapping_name) != 1:
                raise AttributeError(
                    f"No mapping could be inferred between {molecule} and {mapping_name}. Please specify a mapping name!"
                )
            mapping_name = mapping_name[0]
        return mapping_name

    def get_mapping_partner(self, molecule: str, mapping: str) -> str:
        return self.infer_mapping(molecule=molecule, mapping=mapping)

    def get_mapping(
        self,
        mapping_name: str,
        molecule: str = None,
        molecule_columns: List[str] = [],
        partner_columns: List[str] = [],
        partner_molecule: str = None,
    ) -> MoleculeMapping:
        if molecule is not None and molecule not in self.molecules:
            raise KeyError(f"Molecule type {molecule} does not exist!")
        mapping_name = self.infer_mapping_name(molecule=molecule, mapping_name=mapping_name)
        mapping = self.mappings[mapping_name]
        if molecule is not None:
            if molecule != mapping.mapping_molecules[0]:
                mapping = mapping.swaplevel()
            else:
                mapping = mapping.copy()
        else:
            mapping = mapping.copy()
            molecule = mapping.mapping_molecules[0]
        assert molecule == mapping.mapping_molecules[0] and (
            partner_molecule is None or partner_molecule == mapping.mapping_molecules[1]
        )
        molecule, partner_molecule = mapping.mapping_molecules
        if molecule is not None:
            partner = [n for n in mapping.mapping_molecules if n != molecule]
            partner = partner[0]
            if partner_molecule is not None:
                if partner_molecule != partner:
                    raise AttributeError(
                        f"The mapping you specified maps {mapping.mapping_molecules}."
                        + f" This does not match the molecules ({molecule, partner_molecule}) you specified."
                    )
        mol_vals = self.molecules[molecule].loc[mapping.df.index.get_level_values(0), molecule_columns]
        for mc in mol_vals:
            mc = mol_vals[mc]
            mapping.df[mc.name] = mc.values
        mol_vals = self.molecules[partner_molecule].loc[mapping.df.index.get_level_values(1), partner_columns]
        for mc in mol_vals:
            mc = mol_vals[mc]
            mapping.df[mc.name] = mc.values
        return mapping

    def get_mapped(
        self,
        mapping: str,
        molecule: str = None,
        molecule_columns: List[str] = [],
        partner_columns: List[str] = [],
        partner_molecule: str = None,
    ) -> pd.DataFrame:
        mapping = self.get_mapping(
            mapping_name=mapping,
            molecule=molecule,
            molecule_columns=molecule_columns,
            partner_columns=partner_columns,
            partner_molecule=partner_molecule,
        )
        return mapping.df

    def get_mapping_degrees(
        self,
        molecule: str,
        mapping: str,
        result_column: Optional[str] = None,
        partner_molecule: str = None,
        only_unique: bool = False,
    ):
        molecule, mapping, partner = self.infer_mapping(molecule=molecule, mapping=mapping)
        if partner_molecule is not None:
            assert partner_molecule == partner
        else:
            partner_molecule = partner
        mapped = self.get_mapped(mapping=mapping, molecule=molecule, partner_molecule=partner_molecule)
        res = pd.Series(data=0, index=self.molecules[molecule].index)
        mapped["deg"] = 1
        if only_unique:
            if molecule == partner_molecule:
                raise AttributeError(
                    "Only_unique not supported for mappings between only one molecule type are not supported!"
                )
            mapped["partner_deg"] = 1
            partner_degs = mapped.groupby(partner_molecule)["partner_deg"].count()
            mapped[mapped.index.get_level_values(partner_molecule).isin(partner_degs[partner_degs > 1].index)] = 0
        degs = mapped.groupby(molecule)["deg"].sum()
        res.loc[degs.index] = degs
        if result_column is not None:
            self.molecules[molecule][result_column] = res
        return res

    def get_mapping_unique_molecules(
        self, molecule: str, mapping: str = "gene", partner_molecule: Optional[str] = None
    ):
        degs = self.get_mapping_degrees(molecule=molecule, partner_molecule=partner_molecule, mapping=mapping)
        return degs[degs == 1].index

    def set_molecule_data(self, molecule: str, column: str, data: pd.Series, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        self.molecules[molecule][column] = data

    def rename_molecule(self, molecule: str, new_name: str):
        if new_name in self.molecules:
            raise KeyError(f"Key {new_name} already exists.")
        self.molecules[new_name] = self.molecules[molecule]
        affected_mappings = list(chain(*self.mappings_lookup[molecule].values()))
        for m in affected_mappings:
            m = self.mappings[m]
            m.rename_molecule(molecule=molecule, new_name=new_name)
        del self.molecules[molecule]
        self._update_mapping_lookup()

    def drop_mapping(self, mapping: str):
        del self.mappings[mapping]
        self._update_mapping_lookup()

    def rename_mapping(self, mapping: str, new_name: str):
        if new_name in self.mappings:
            raise KeyError(f"Key {new_name} already exists.")
        self.mappings[new_name] = self.mappings[mapping]
        del self.mappings[mapping]
        self._update_mapping_lookup()

    def rename_molecule_data(self, columns: Dict[str, str], molecule: Optional[str] = None, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        if molecule is None:
            molecules = self.molecule_names
        else:
            molecules = [molecule]
        for molecule in molecules:
            self.molecules[molecule].rename(columns=columns, inplace=True)

    def drop_molecule_data(self, columns: List[str], molecule: Optional[str] = None, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        if molecule is None:
            molecules = self.molecule_names
        else:
            molecules = [molecule]
        for molecule in molecules:
            self.molecules[molecule].drop(columns=columns)

    def add_molecules(self, molecule: str, df: pd.DataFrame):
        if molecule not in self.molecules:
            self.molecules[molecule] = df
        else:
            if df.index.isin(self.molecules[molecule].index).any():
                raise AttributeError("At least one of the provided molecule indices does already exist.")
            df.index.name = "id"
            self.molecules[molecule] = pd.concat([self.molecules[molecule], df])
        self.clear_cache()

    def _update_mapping_lookup(self):
        self.mappings_lookup = {}
        for mapping_name, mapping in self.mappings.items():
            molecules = mapping.mapping_molecules
            lookup = self.mappings_lookup.get(molecules[0], dict())
            lookup[molecules[1]] = lookup.get(molecules[1], []) + [mapping_name]
            self.mappings_lookup[molecules[0]] = lookup
            lookup = self.mappings_lookup.get(molecules[1], dict())
            lookup[molecules[0]] = lookup.get(molecules[0], []) + [mapping_name]
            self.mappings_lookup[molecules[1]] = lookup

    def add_mapping_pairs(self, name: str, pairs: pd.DataFrame, mapping_molecules: Optional[Tuple[str, str]]):
        if name not in self.mappings:
            mapping = MoleculeMapping(name=name, df=pairs, mapping_molecules=mapping_molecules)
            mapping.validate_for_molecule_set(molecule_set=self)
            self.mappings[name] = mapping
            self._update_mapping_lookup()
        else:
            mols = self.mappings[name].mapping_molecules
            if mapping_molecules is None:
                mapping_molecules = tuple(pairs.index.names)
            if set(mols) != set(mapping_molecules):
                raise AttributeError("The index of the provided dataframe does not match the existing mapping.")
            if mols != mapping_molecules:
                pairs = pairs.swaplevel()
            self.mappings[name].add_pairs(pairs)
            self.mappings[name].validate_for_molecule_set(self)
        self.clear_cache()

    def create_graph(self, mapping: str = "gene", bidirectional: bool = True, cache: bool = True) -> MoleculeGraph:
        if cache and mapping in self.graphs:
            return self.graphs[mapping]
        nodes, edges, node_mapping, type_mapping = create_graph_nodes_edges(
            molecule_set=self, mappings=[mapping], make_bidirectional=bidirectional
        )
        graph = MoleculeGraph(
            nodes=nodes, edges=edges, node_mapping=node_mapping, molecule_set=self, type_mapping=type_mapping
        )
        if cache:
            self.graphs[mapping] = graph
        return graph

    def get_node_values_for_graph(self, graph: MoleculeGraph, include_id_and_type: bool = True):
        node_values = []
        for node_type, df in graph.nodes.groupby("type"):
            key = graph.inverse_type_mapping[node_type]  # type: ignore
            values = self.molecules[key]
            columns = list(values.columns)
            df[columns] = values.loc[df.molecule_id, columns].values
            if include_id_and_type:
                node_values.append(df)
            else:
                node_values.append(df.loc[:, columns])
        node_values = pd.concat(node_values)
        return node_values

    # def create_graph_dgl(self, mapping:str = 'gene')->dgl.DGLGraph:
    #     nodes, edges, node_mapping = self.create_graph_nodes_edges(bidirectional=True)
    #     return create_graph_dgl(nodes=nodes, edges=edges)

    # @property
    # def nodes(self):
    #     if self._nodes is None:
    #         self.create_graph_nodes_edges()
    #     return self._nodes

    # @property
    # def edges(self):
    #     if self._edges is None:
    #         self.create_graph_nodes_edges()
    #     return self._edges

    # @property
    # def node_mapping(self) -> Dict[str, pd.DataFrame]:
    #     if self._node_mapping is None:
    #         self.create_graph_nodes_edges()
    #     return self._node_mapping # type: ignore
