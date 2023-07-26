from typing import Dict, Optional, List, Iterable, Union
import uuid
from pathlib import Path

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

    def __init__(self, molecules: Dict[str, pd.DataFrame], mappings: Dict[str, Dict[str, pd.DataFrame]]):
        """A set of molecules and their relations/mapping.

        Args:
            molecules (Dict[str, pd.DataFrame]): Relevant molecules as dictionay of molecule name and dataframe containing at least the molecule indices
            mappings (Dict[str, Dict[str, pd.DataFrame]]): Dictionary of molecules mappings.
                Every mapping is a dictionary of molecule name and mapping as dataframe containing mapping id and molecule index pairs
        """
        self.molecules = molecules
        self.mappings = mappings
        self.graphs = dict()
        # self._node_mapping: Optional[Dict[str, pd.DataFrame]] = None
        # self._nodes = None
        # self._edges = None
        self.id = str(uuid.uuid4())

    # def create_graph_nodes_edges(self, mapping:str = 'gene', bidirectional: bool = True):
    #     if self._nodes is None or self._edges is None or self._node_mapping is None:
    #         self._nodes, self._edges, self._node_mapping = create_graph_nodes_edges(molecule_set=self, mapping=mapping, make_bidirectional=bidirectional)
    #     return self._nodes, self._edges, self._node_mapping

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
                _, _, mapping, molecule = key.split("/")
                if mapping not in mappings:
                    mappings[mapping] = {}
                mappings[mapping][molecule] = store[key]
        return cls(molecules=molecules, mappings=mappings)

    def save(self, path: Union[str, Path], overwrite: bool = False):
        path = Path(path)
        if not overwrite and path.exists():
            raise RuntimeError(f"{path} already exists!")
        with HDFStore(path) as store:
            for molecule, df in self.molecules.items():
                _check_name(molecule)
                store[f"molecule/{molecule}"] = df
            for mapping_name, mapping_dict in self.mappings.items():
                _check_name(mapping_name)
                for molecule, df in mapping_dict.items():
                    store[f"mapping/{mapping_name}/{molecule}"] = df

    def copy(self) -> "MoleculeSet":
        molecules = {}
        for n, v in self.molecules.items():
            molecules[n] = v.copy()
        mappings = {}
        for mapping_name, mapping in self.mappings.items:
            mapping_copy = {}
            for n, v in mapping.items():
                mapping_copy[n] = v.copy()
            mappings[mapping_name] = mapping_copy
        return MoleculeSet(molecules=molecules, mappings=mapping)

    def number_molecules(self, molecule: str) -> int:
        return len(self.molecules[molecule])

    def get_mapped_pairs(self, molecule_a: str, molecule_b: str, mapping: str = "gene"):
        sources = self.mappings[mapping][molecule_a].rename(columns={"id": molecule_a})
        destinations = self.mappings[mapping][molecule_b].rename(columns={"id": molecule_b})
        return sources.merge(destinations, on="map_id", how="inner")

    def get_mapping_degrees(self, molecule: str, partner_molecule: str, mapping: str = "gene", result_column: Optional[str] = None):
        mapped = self.get_mapped_pairs(molecule_a=molecule, molecule_b=partner_molecule, mapping=mapping)
        res = pd.Series(data=0, index=self.molecules[molecule].index)
        degs = mapped.groupby(molecule)[partner_molecule].count()
        res.loc[degs.index] = degs
        if result_column is not None:
            self.molecules[molecule][result_column] = res
        return res

    def get_mapping_unique_molecules(self, molecule: str, partner_molecule: str, mapping: str = "gene"):
        degs = self.get_mapping_degrees(molecule=molecule, partner_molecule=partner_molecule, mapping=mapping)
        return degs[degs == 1].index

    def set_molecule_data(self, molecule: str, column: str, data: pd.Series, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        self.molecules[molecule][column] = data

    def rename_molecule_data(self, columns: Dict[str, str], molecule: Optional[str] = None, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        if molecule is None:
            molecules = self.molecule_names
        else:
            molecules = [molecule]
        for molecule in molecules:
            self.molecules[molecule].rename(columns=columns)

    def drop_molecule_data(self, columns: List[str], molecule: Optional[str] = None, inplace: bool = True):
        if inplace == False:
            raise NotImplementedError()
        if molecule is None:
            molecules = self.molecule_names
        else:
            molecules = [molecule]
        for molecule in molecules:
            self.molecules[molecule].drop(columns=columns)

    def create_graph(self, mapping: str = "gene", bidirectional: bool = True, cache: bool = True) -> MoleculeGraph:
        if cache and mapping in self.graphs:
            return self.graphs[mapping]
        nodes, edges, node_mapping, type_mapping = create_graph_nodes_edges(
            molecule_set=self, mapping=mapping, make_bidirectional=bidirectional
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
