from typing import List

import dgl
import torch

from ..data.masked_dataset import MaskedDataset


def masked_dataset_to_homogeneous_graph(masked_datasets: List[MaskedDataset], mappings: List[str], target:str, features:List[str]=[])->dgl.DGLGraph:
    graphs = []
    for masked_dataset in masked_datasets:
        graph = masked_dataset.to_dgl_graph(molecule_features={mol:[target] + features for mol in masked_dataset.dataset.molecules.keys()},
                                             mappings=mappings)
        graphs.append(graph)
    graphs = masked_heterograph_to_homogeneous(masked_heterographs=graphs, target=target, features=features)
    return graphs

def masked_heterograph_to_homogeneous(masked_heterographs: List[dgl.DGLGraph], target:str, features:List[str]=[])->List[dgl.DGLGraph]:
    res = []
    for graph in masked_heterographs:
        graph = dgl.to_homogeneous(graph, ndata= [target] + features + ['mask', 'hidden'])
        molecule_type = torch.nn.functional.one_hot(graph.ndata[dgl.NTYPE])
        features = [graph.ndata[f] for f in features]
        x = torch.concat(features + [molecule_type], axis=-1)
        graph.ndata['features'] = x
        graph.ndata['target'] = graph.ndata[target]
        pop_keys = []
        for key in graph.ndata.keys():
            if key not in {'features', 'target', 'mask', 'hidden', dgl.NID, dgl.NTYPE, dgl.EID, dgl.ETYPE}:
                pop_keys.append(key)
            elif key not in {'mask', 'hidden'}:
                graph.ndata[key] = graph.ndata[key].type(torch.float32)
        for key in pop_keys:
            graph.ndata.pop(key)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)
        res.append(graph)
    return res