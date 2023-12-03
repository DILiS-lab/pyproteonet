from typing import Dict, Optional, List, Iterable, Union, Tuple

import pandas as pd
import numpy as np
import torch
import dgl

from .dataset import Dataset
from .dataset_sample import DatasetSample
from .molecule_graph import MoleculeGraph
from .abstract_masked_dataset import AbstractMaskedDataset
from ..dgl.graph_key_dataset import GraphKeyDataset

class MaskedDataset(AbstractMaskedDataset):
    def __init__(
        self,
        dataset: Dataset,
        masks: Dict[str, pd.DataFrame],
        hidden: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.dataset = dataset
        self._keys = set.union(*[set(m.keys()) for m in masks.values()])
        self.masks = masks
        self.hidden = dict()
        if hidden is not None:
            self.hidden = hidden
            self._keys = set.union(self._keys, *[set(m.keys()) for m in hidden.values()])
        self._keys = list(self._keys)

    @classmethod
    def from_ids(
        cls, dataset: Dataset, mask_ids: Dict[str, pd.Index], hidden_ids: Optional[Dict[str, pd.Index]] = None
    ) -> "MaskedDataset":
        masks = dict()
        for mol, ids in mask_ids.items():
            masks[mol] = _ids_to_mask(dataset=dataset, molecule=mol, ids=ids)
        hidden = None
        if hidden_ids is not None:
            hidden = dict()
            for mol, ids in hidden_ids.items():
                hidden[mol] = _ids_to_mask(dataset=dataset, molecule=mol, ids=ids)
        return cls(dataset=dataset, masks=masks, hidden=hidden)
    
    def set_mask(self, molecule: str, mask: pd.DataFrame) -> None:
        self.masks[molecule] = mask

    def get_mask_ids(self, molecule: str) -> pd.Index:
        ids = self.masks[molecule].stack().swaplevel()
        ids = ids[ids]
        return ids.index.set_names(['sample','id'])

    def set_mask_ids(self, molecule: str, ids: pd.Index) -> None:
        self.masks[molecule] = _ids_to_mask(dataset=self.dataset, molecule=molecule, ids=ids)

    def set_hidden(self, molecule: str, hidden: pd.DataFrame) -> None:
        self.hidden[molecule] = hidden

    def get_hidden_ids(self, molecule: str) -> pd.Index:
        ids = self.hidden[molecule].stack().swaplevel()
        ids = ids[ids]
        return ids.index.set_names(['sample','id'])

    def set_hidden_ids(self, molecule: str, ids: pd.Index) -> None:
        self.hidden[molecule] = _ids_to_mask(dataset=self.dataset, molecule=molecule, ids=ids)

    def keys(self) -> Iterable[str]:
        return self._keys

    @property
    def has_hidden(self) -> bool:
        if len(self.hidden):
            return True
        return False

    def get_sample(self, key: str) -> DatasetSample:
        return self.dataset[key]

    def get_masked_nodes(self, key: str, graph: MoleculeGraph) -> Iterable[int]:
        res = [[]]
        for mol, mask in self.masks.items():
            node_mapping = graph.node_mapping[mol]
            mask_nodes = mask[key]
            mask_nodes = mask_nodes.loc[mask_nodes].index
            res.append(node_mapping.loc[mask_nodes, "node_id"].to_numpy())  # type: ignore
        return np.concatenate(res)

    def set_samples_value_matrix(self, matrix: Union[np.array, pd.DataFrame], molecule: str, column: str, only_set_masked: bool = True) -> None:
        mol_ids = self.dataset.molecules[molecule].index
        mat_df = self.dataset.get_samples_value_matrix(molecule=molecule, column=column)
        mat_df = mat_df.loc[mol_ids]
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values
        if only_set_masked:
            mask = self.masks[molecule].loc[mol_ids].values
            mat_df.values[mask] = matrix[mask]
        else:
            mat_df.values[:,:] = matrix
        self.dataset.set_samples_value_matrix(matrix=mat_df, molecule=molecule, column=column)

    def get_hidden_nodes(self, key: str, graph: MoleculeGraph) -> Iterable[int]:
        if self.hidden is None or key not in self.hidden.columns:
            return []
        res = [[]]
        for mol, mask in self.hidden.items():
            node_mapping = graph.node_mapping[mol]
            hidden_nodes = mask[key]
            hidden_nodes = hidden_nodes.loc[hidden_nodes].index
            res.append(node_mapping.loc[hidden_nodes, "node_id"].to_numpy())  # type: ignore
        return np.concatenate(res)

    def get_graph_dataset_dgl(
        self,
        mapping: str = "gene",
        value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
        molecule_columns: List[str] = [],
        target_column: str = "abundance",
        missing_column_value: Optional[float] = None,
    ) -> GraphKeyDataset:
        return GraphKeyDataset(
            masked_dataset=self,
            mapping=mapping,
            value_columns=value_columns,
            molecule_columns=molecule_columns,
            target_column=target_column,
            missing_column_value=missing_column_value,
        )

    def to_dgl_graph(self, molecule_features: Dict[str, Union[str, List[str]]], mappings: Union[str, List[str]], mapping_directions: Dict[str, Tuple[str, str]] = {},
                     features_to_float32: bool=True) -> dgl.DGLHeteroGraph:
        graph_data = dict()
        if isinstance(mappings, str):
            mappings = [mappings]
        for mapping_name in mappings:
            mapping = self.dataset.mappings[mapping_name]
            if mapping_name in mapping_directions:
                if tuple(mapping_directions[mapping_name]) != mapping.mapping_molecules:
                    mapping = mapping.swaplevel()
            identifier = (mapping.mapping_molecules[0], mapping_name, mapping.mapping_molecules[1])
            edges = []
            for i, mol in enumerate(mapping.mapping_molecules):
                e_data = self.dataset.molecules[mol].index.get_indexer(mapping.df.index.get_level_values(i))
                edges.append(torch.from_numpy(e_data))
            edges = tuple(edges)
            graph_data[identifier] = edges
        g = dgl.heterograph(graph_data)
        num_samples = len(self.dataset.sample_names)
        for mol, mol_features in molecule_features.items():
            if isinstance(mol_features, str):
                mol_features = [mol_features]
            mol_ids = self.dataset.molecules[mol].index
            for feature in mol_features:
                if feature in {'hidden', 'mask'}:
                    raise KeyError('Feature names "hidden" and "mask" are reserved names')
                mat = self.dataset.get_samples_value_matrix(molecule=mol, column=feature).loc[mol_ids]
                feat = torch.from_numpy(mat.to_numpy())
                if features_to_float32:
                    feat = feat.to(torch.float32)
                g.nodes[mol].data[feature] = feat
            if mol in self.masks:
                g.nodes[mol].data['mask'] = torch.from_numpy(self.masks[mol].loc[mol_ids].to_numpy())
            else:
                g.nodes[mol].data['mask'] = torch.full((mol_ids.shape[0], num_samples), False)
            if mol in self.hidden:
                g.nodes[mol].data['hidden'] = torch.from_numpy(self.hidden[mol].loc[mol_ids].to_numpy())
            else:
                g.nodes[mol].data['hidden'] = torch.full((mol_ids.shape[0], num_samples), False)
        return g

def _ids_to_mask(dataset: Dataset, molecule: str, ids: pd.Index):
    mask = pd.DataFrame(index=dataset.molecules[molecule].index, data={sample: False for sample in dataset.sample_names})
    if "sample" in ids.names:
        m = pd.Series(index=ids, data=True)
        m = m.unstack(level="sample", fill_value=False)
        mask.loc[m.index, m.columns] = m
    else:
        for sample in dataset.sample_names:
            mask.loc[ids, sample] = True
    return mask
