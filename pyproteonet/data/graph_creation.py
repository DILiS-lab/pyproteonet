import itertools
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING, Iterable, Union
import logging

import pandas as pd
import numpy as np
import dgl
import torch

if TYPE_CHECKING:
    from .molecule_set import MoleculeSet
    from .dataset_sample import DatasetSample
    from .molecule_graph import MoleculeGraph

NODE_TYPE_MAPPING = {
    "peptide": 0,
    "mRNA": 1,
    "protein": 2,
}
NODE_TYPE_INVERSE_MAPPING = {v: k for k, v in NODE_TYPE_MAPPING.items()}

logger = logging.Logger("graph_creation")


def create_graph_nodes_edges(
    molecule_set: "MoleculeSet", mapping: str = "gene", make_bidirectional: bool = True, add_self_edges: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, int]]:
    node_mapping = dict()
    offset = 0
    for key in molecule_set.molecules.keys():
        index = molecule_set.molecules[key].index
        size = index.shape[0]
        node_ids = np.arange(offset, offset + size)
        offset += size
        node_mapping[key] = pd.DataFrame({"node_id": node_ids}, index=index)
    edges = []
    for source_key, destination_key in itertools.combinations(molecule_set.molecules.keys(), 2):
        sources = molecule_set.mappings[mapping][source_key].copy()
        sources["source_node"] = node_mapping[source_key].loc[sources.id, "node_id"].to_numpy()
        destinations = molecule_set.mappings[mapping][destination_key].copy()
        destinations["destination_node"] = node_mapping[destination_key].loc[destinations.id, "node_id"].to_numpy()
        edges.append(sources.merge(destinations, on="map_id", how="inner").loc[:, ["source_node", "destination_node"]])
    edges = pd.concat(edges, ignore_index=True)
    if make_bidirectional:
        edges2 = pd.DataFrame({"source_node": edges.destination_node, "destination_node": edges.source_node})
        edges = pd.concat([edges, edges2], ignore_index=True)
    nodes = []
    for key in node_mapping.keys():
        nm = node_mapping[key].copy()
        nm["molecule_id"] = nm.index
        nm.set_index("node_id", inplace=True, verify_integrity=True)
        nm["type"] = NODE_TYPE_MAPPING[key]
        nm["type"] = nm["type"].astype(int)
        nodes.append(nm)
        # assert ('node_id' not in a.columns and 'type' not in a.columns
        # a['node_id'] = node_mapping[key].loc[a.index, 'node_id']
        # a['type'] =
    nodes = pd.concat(nodes, verify_integrity=True)
    if add_self_edges:
        self_edges = pd.DataFrame({"source_node": np.arange(len(nodes)), "destination_node": np.arange(len(nodes))})
        edges = pd.concat([edges, self_edges], ignore_index=True)
    return nodes, edges, node_mapping, NODE_TYPE_MAPPING


def create_graph_dgl(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    external_edges: Optional[pd.DataFrame] = None,
    edge_name: str = "interacts",
    add_self_loops: bool = False,
) -> dgl.DGLGraph:
    if external_edges != None:
        raise NotImplemented("External edges are not supported in this method yet")
    graph_data = {
        ("molecule", edge_name, "molecule"): (edges.source_node, edges.destination_node),
    }
    # ('molecule', 'external', 'molecule'): ([], [])}
    dgl_N = dgl.heterograph(graph_data, num_nodes_dict={"molecule": nodes.shape[0]})  # type: ignore
    # dgl_N = dgl.graph(data = (edges.source_node, edges.destination_node), num_nodes = nodes.shape[0])
    if "weight" in edges.columns:
        dgl_N.edges[edge_name].data["w"] = torch.tensor(edges.weight.to_numpy().astype(np.float32))
    else:
        dgl_N.edges[edge_name].data["w"] = torch.ones((edges.shape[0],), dtype=torch.float32)
    if add_self_loops:
        dgl_N = dgl.add_self_loop(dgl_N, etype=edge_name)
    return dgl_N


def populate_graph_dgl(
    graph: "MoleculeGraph",
    dgl_graph: dgl.DGLGraph,
    dataset_sample: "DatasetSample",
    value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
    molecule_columns: List[str] = [],
    target_column: str = "abundance",
):
    if not (isinstance(value_columns, list) or isinstance(value_columns, dict)):
        raise ValueError(
            "value_columns must either be list of strings representing the colums present " +
            "for any molecule to use as graph features or a dict mapping from molecule type name" +
            "to list of stings representing the value columns to use for every molecule type."
        )
    if isinstance(value_columns, list):
        node_type_groups = [(node_type, df) for node_type, df in graph.nodes.groupby("type")]
        value_columns = {graph.inverse_type_mapping[node_type]: value_columns for node_type,_ in node_type_groups} # type: ignore
    all_value_columns = []
    for vcs in value_columns.values():
        all_value_columns.extend(vcs)
    if target_column not in molecule_columns and target_column not in all_value_columns:
        raise ValueError("neither any value column nor any molecule column equals the target column!")
    num_value_columns = max([len(columns) for _,columns in value_columns.items()])
    node_molecule_values = dataset_sample.molecule_set.get_node_values_for_graph(graph=graph, include_id_and_type=False)
    num_nodes = dgl_graph.num_nodes("molecule")
    num_columns = num_value_columns + len(molecule_columns)
    x = np.full((num_nodes, num_columns), dataset_sample.missing_abundance_value, dtype=np.float32)
    target = np.full((num_nodes, 1), dataset_sample.missing_abundance_value, dtype=np.float32)
    for molecule, columns in value_columns.items():
        nodes = graph.node_mapping[molecule].loc[dataset_sample.values[molecule].index, 'node_id']
        for i, column in enumerate(columns):
            values = dataset_sample.values[molecule].loc[:, column].to_numpy().astype(np.float32)
            x[nodes, i] = values
            if column == target_column:
                target[nodes, i] = values
    # for molecule, columns in value_columns.item():
    #     x[node_values.index, i] = node_values[column].to_numpy().astype(np.float32)
    #     if column == target_column:
    #         target[node_values.index, i] = node_values[column].to_numpy().astype(np.float32)
    #     i += 1
    for i, column in enumerate(molecule_columns):
        i = i + num_value_columns
        values = node_molecule_values[column].to_numpy().astype(np.float32)
        x[node_molecule_values.index, i] = values
        if column == target_column:
            target[node_molecule_values.index, i] = values
        i += 1
    nodes_data = graph.nodes.loc[:, ("type",)] # type: ignore
    # nodes_data[node_values.columns] = node_values
    node_molecule_labels = np.zeros((nodes_data.shape[0], int(nodes_data.max() + 1)), dtype=np.float32)
    node_molecule_labels[nodes_data.index.to_numpy(), nodes_data["type"].to_numpy()] = 1
    # dgl_N = dgl.to_bidirected(dgl_N)
    dgl_graph.nodes["molecule"].data["x"] = torch.from_numpy(np.append(x, node_molecule_labels, axis=1))
    dgl_graph.nodes["molecule"].data["target"] = torch.from_numpy(target)
    # for i, column in enumerate(value_columns):
    #    dgl_graph.nodes['molecule'].data[column] = torch.from_numpy(x[:, i]).unsqueeze(axis=1)
    dgl_graph.nodes["molecule"].data["type"] = torch.from_numpy(node_molecule_labels)

    # dgl_N.edges['external'].data['w'] = torch.tensor(edge_weights_external.tolist())
    # dgl_graph = dgl.add_self_loop(dgl_graph, etype = 'external')

    # dgl.save_graphs(save_sample_name, dgl_N)
    logger.info("node abundances", x.shape)
    logger.info("number of nodes DGL", dgl_graph.num_nodes())
    logger.info("number of edges DGL", dgl_graph.num_edges())
    # print('Node Data', dgl_N.ndata['x'])
    # print('Edge Data', dgl_N.edata['w'])


def populate_graph_dgl_deprecated(
    graph: "MoleculeGraph",
    dgl_graph: dgl.DGLGraph,
    dataset_sample: "DatasetSample",
    value_columns: Union[Dict[str, List[str]], List[str]] = ["abundance"],
    molecule_columns: List[str] = [],
):
    if not isinstance(value_columns, list) or isinstance(value_columns, dict):
        raise ValueError(
            "value_columns must either be list of strings representing the colums present " +
            "for any molecule to use as graph features or a dict mapping from molecule type name" +
            "to list of stings representing the value columns to use for every molecule type."
        )
    node_type_groups = [(node_type, df) for node_type, df in graph.nodes.groupby("type")]
    if isinstance(value_columns, list):
        value_columns = {node_type: value_columns for node_type,_ in node_type_groups} # type: ignore
    else:
        value_columns = {graph.type_mapping[molecule]: columns for molecule, columns in value_columns} # type: ignore
    node_values = dataset_sample.get_node_values_for_graph(graph=graph, include_id_and_type=False)
    num_nodes = dgl_graph.num_nodes("molecule")
    x = np.full((num_nodes, len(value_columns)), dataset_sample.missing_abundance_value, dtype=np.float32)
    for i, column in enumerate(value_columns):
        x[node_values.index, i] = node_values[column].to_numpy().astype(np.float32)
    nodes_data = graph.nodes.loc[:, ("type",)]
    # nodes_data[node_values.columns] = node_values
    node_molecule_labels = np.zeros((nodes_data.shape[0], int(nodes_data.max() + 1)), dtype=np.float32)
    node_molecule_labels[nodes_data.index.to_numpy(), nodes_data["type"].to_numpy()] = 1

    # dgl_N = dgl.to_bidirected(dgl_N)
    dgl_graph.nodes["molecule"].data["x"] = torch.from_numpy(np.append(x, node_molecule_labels, axis=1))
    for i, column in enumerate(value_columns):
        dgl_graph.nodes["molecule"].data[column] = torch.from_numpy(x[:, i]).unsqueeze(axis=1)
    dgl_graph.nodes["molecule"].data["type"] = torch.from_numpy(node_molecule_labels)

    if "label" in node_values.columns:
        node_abundance_labels = np.full((num_nodes, 1), dataset_sample.missing_label_value, dtype=np.float32)
        node_abundance_labels[node_values.index, 0] = node_values["label"].to_numpy().astype(np.float32)
        logger.info("node_abundance_labels", "node_molecule_labels")
        logger.info(node_abundance_labels.shape, node_molecule_labels.shape)
        dgl_graph.nodes["molecule"].data["labels"] = torch.from_numpy(node_abundance_labels)

    if "weight" in graph.edges.columns:
        dgl_graph.edges["interacts"].data["w"] = torch.tensor(graph.edges.weight.to_numpy().astype(np.float32))
    else:
        dgl_graph.edges["interacts"].data["w"] = torch.ones((graph.edges.shape[0],), dtype=torch.float32)
    # dgl_N.edges['external'].data['w'] = torch.tensor(edge_weights_external.tolist())
    dgl_graph = dgl.add_self_loop(dgl_graph, etype="external")
    dgl_graph = dgl.add_self_loop(dgl_graph, etype="interacts")

    # dgl.save_graphs(save_sample_name, dgl_N)
    logger.info("node abundances", x.shape)
    logger.info("number of nodes DGL", dgl_graph.num_nodes())
    logger.info("number of edges DGL", dgl_graph.num_edges())
    # print('Node Data', dgl_N.ndata['x'])
    # print('Edge Data', dgl_N.edata['w'])
