from typing import List, Optional

import dgl
import torch

from ..data.masked_dataset import MaskedDataset


def masked_dataset_to_homogeneous_graph(masked_datasets: List[MaskedDataset], mappings: List[str], target:str, features:List[str]=[],
                                        sample_lists: Optional[List[str]] = None)->dgl.DGLGraph:
    graphs = []
    if sample_lists is None:
        sample_lists = [masked_dataset.dataset.sample_names for masked_dataset in masked_datasets]
    else:
        if len(sample_lists) != len(masked_datasets):
            raise ValueError('sample_lists must have the same length as masked_datasets')
    for masked_dataset, samples in zip(masked_datasets, sample_lists):
        graph = masked_dataset.to_dgl_graph(feature_columns={mol:[target] + features for mol in masked_dataset.dataset.molecules.keys()},
                                            mappings=mappings, samples=samples)
        graphs.append(graph)
    graphs = masked_heterograph_to_homogeneous(masked_heterographs=graphs, target=target, features=features)
    return graphs

def masked_heterograph_to_homogeneous(masked_heterographs: List[dgl.DGLGraph], target:str, features:List[str]=[])->List[dgl.DGLGraph]:
    res = []
    for graph in masked_heterographs:
        graph = dgl.to_homogeneous(graph, ndata= [target] + features + ['mask', 'hidden'])
        molecule_type = torch.nn.functional.one_hot(graph.ndata[dgl.NTYPE])
        x = [graph.ndata[f] for f in features]
        x = torch.concat(x + [molecule_type], axis=-1)
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