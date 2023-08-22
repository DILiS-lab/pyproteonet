from typing import Dict, Optional, List, Iterable, Union
import uuid
from pathlib import Path
import warnings
from itertools import chain

import pandas as pd
from pandas import HDFStore

from .graph_creation import create_graph_nodes_edges
from .molecule_graph import MoleculeGraph


def _check_name(name: str):
    if "/" in name:
        raise KeyError('Names must not include "/"!')


class MoleculeSet:
    """A set of molecules and their relations/mapping.
    E.g. a set of proteins and peptides with every peptide belonging to one or many proteins
    """

    def __init__(self, molecules: Dict[str, pd.DataFrame], mappings: Dict[str, pd.DataFrame]):
        """A set of molecules and their relations/mapping.

        Args:
            molecules (Dict[str, pd.DataFrame]): Relevant molecules as dictionay of molecule name and dataframe containing at least the molecule indices
            mappings (Dict[str, pd.DataFrame]): Every mapping has a multi-index, where every item consists of the ids of the two molecues that are mapped
        """
        self.molecules = molecules
        self.mappings = {}
        self.mappings_lookup = {}
        for mapping_name, mapping in mappings.items():
            self._validate_mapping(mapping)
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

    def _validate_mapping(self, mapping: pd.DataFrame, mapping_name: Optional[str] = None):
        assert len(mapping.index.names) == 2
        assert all([m in self.molecules for m in mapping.index.names])
        if mapping_name in self.molecules:
            warnings.warn(f"Mapping '{mapping_name}' has the same name as a molecule. This can lead to ambiguity and should be avoided!")
        mol_a, mol_b = mapping.index.names
        assert mapping.index.get_level_values(mol_a).isin(self.molecules[mol_a].index).all()
        assert mapping.index.get_level_values(mol_b).isin(self.molecules[mol_b].index).all()
        assert mapping.index.is_unique

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
                mappings[key.split("/")[-1]] = store[key]
        return cls(molecules=molecules, mappings=mappings)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        if not overwrite and path.exists():
            raise RuntimeError(f"{path} already exists!")
        with HDFStore(path) as store:
            for molecule, df in self.molecules.items():
                _check_name(molecule)
                store[f"molecule/{molecule}"] = df
            for mapping_name, df in self.mappings.items():
                _check_name(mapping_name)
                store[f"mapping/{mapping_name}"] = df

    def copy(self) -> "MoleculeSet":
        molecules = {}
        for n, v in self.molecules.items():
            molecules[n] = v.copy()
        mappings = {}
        for mapping_name, mapping in self.mappings.items():
            mappings[mapping_name] = mapping.copy()
        return MoleculeSet(molecules=molecules, mappings=mappings)

    def number_molecules(self, molecule: str) -> int:
        return len(self.molecules[molecule])

    def get_mapped_pairs(self, mapping: str, molecule_a: str = None, molecule_b: str = None):
        mapping = self.mappings[mapping]
        if molecule_a is not None and molecule_b is not None:
            assert molecule_a in mapping.index.names and molecule_b in mapping.index.names
        if len(mapping.columns):
            mapping = mapping.loc[:, []]
        mapping = mapping.reset_index(drop=False)  # TODO: don't do this once everything is refactored
        return mapping

    def get_mapped(
        self,
        mapping: str,
        molecule: str = None,
        molecule_columns: List[str] = [],
        molecule_columns_partner: List[str] = [],
        partner_molecule: str = None,
    ):
        if mapping not in self.mappings and molecule is not None:
            mapping = self.mappings_lookup[molecule][mapping]
            if len(mapping) != 1:
                raise AttributeError(f"No mapping could be inferred between {molecule} and {mapping}. Please specify a mapping name!")
            mapping = mapping[0]
        mapped = self.mappings[mapping].copy()
        if molecule is not None:
            partner = [n for n in mapped.index.names if n!= molecule]
            assert len(partner) == 1
            partner = partner[0]
            if partner_molecule is not None:
                if partner_molecule != partner:
                    raise AttributeError(f"The mapping you specified maps {list(mapped.index.names)}." +
                                         f" This does not match the molecules ({molecule, partner_molecule}) you specified.")
        if molecule is not None and partner_molecule is not None:
            assert molecule in mapped.index.names and partner_molecule in mapped.index.names
        else:
            molecule, partner_molecule = mapped.index.names
        mol_vals = self.molecules[molecule].loc[mapped.index.get_level_values(molecule), molecule_columns]
        for mc in mol_vals:
            mapped[mc.name] = mc.values
        mol_vals = self.molecules[partner_molecule].loc[
            mapped.index.get_level_values(partner_molecule), molecule_columns_partner
        ]
        for mc in mol_vals:
            mapped[mc.name] = mc.values
        return mapped

    def get_mapping_degrees(
        self, molecule: str, mapping: str, result_column: Optional[str] = None, partner_molecule: str = None
    ):
        mapped = self.get_mapped(mapping=mapping, molecule=molecule, partner_molecule=partner_molecule)
        res = pd.Series(data=0, index=self.molecules[molecule].index)
        mapped['deg'] = 1
        degs = mapped.groupby(molecule)['deg'].count()
        res.loc[degs.index] = degs
        if result_column is not None:
            self.molecules[molecule][result_column] = res
        return res

    def get_mapping_unique_molecules(self, molecule: str, mapping: str = "gene", partner_molecule: Optional[str] = None):
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
            m.index.rename(level=molecule, names=new_name, inplace=True)
        del self.molecules[molecule]
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
        for mapping_name, mapping_df in self.mappings.items():
            molecules = mapping_df.index.names
            lookup = self.mappings_lookup.get(molecules[0], dict())
            lookup[molecules[1]] = lookup.get(molecules[1], []) + [mapping_name]
            self.mappings_lookup[molecules[0]] = lookup
            lookup = self.mappings_lookup.get(molecules[1], dict())
            lookup[molecules[0]] = lookup.get(molecules[0], []) + [mapping_name]
            self.mappings_lookup[molecules[1]] = lookup

    def add_mapping_pairs(self, mapping: str, pairs: pd.DataFrame):
        if mapping not in self.mappings:
            self._validate_mapping(mapping=pairs, mapping_name=mapping)
            self.mappings[mapping] = pairs
            self._update_mapping_lookup()
        else:
            mols = self.mappings[mapping].index.names
            if set(mols) != set(pairs.index.names):
                raise AttributeError("The index of the provided dataframe does not match the existing mapping.")
            if mols != pairs.index.names:
                pairs = pairs.swaplevel()
            new_mapping = pd.concat([self.mappings[mapping], pairs])
            self._validate_mapping(mapping=new_mapping, mapping_name=mapping)
            self.mappings[mapping] = new_mapping
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
