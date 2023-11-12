from typing import List

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv, GATv2Conv
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .resettable_module import ResettableModule

class DeepGAT(ResettableModule):
	def __init__(self, in_dim, heads: List[int], gat_dims: List[int], out_dim: int = 1, use_gatv2: bool = False, initial_dense_layers: List[int] = []):
		super().__init__()
		if len(initial_dense_layers) > 0:
			dense_layers = []
			for dim in initial_dense_layers:
				dense_layers.append(nn.Linear(in_dim, dim))
				dense_layers.append(nn.ReLU())
				in_dim = dim
			self.initial_layers = nn.Sequential(*dense_layers)
		else:
			self.initial_layers = nn.Identity()
		layers = []
		assert len(gat_dims) == len(heads)
		layer_type = GATConv
		if use_gatv2:
			layer_type = GATv2Conv
		for i,(d,h) in enumerate(zip(gat_dims, heads)):
			if i == 0:
				last_d = in_dim
				last_h = 1
			else:
				last_d = gat_dims[i-1]
				last_h = heads[i-1]
			layers.append(layer_type(in_feats=last_d*last_h, out_feats=d, num_heads=h))
		self.gat_layers: List[layer_type] = nn.ModuleList(layers)
		self.out_layer = layer_type(in_feats=gat_dims[-1]*heads[-1], out_feats=out_dim, num_heads=1)
		
	def reshape_multihead_output(self, h):
		h_concat = []
		for h_idx in range(h.size()[1]):
			h_concat.append(h[:, h_idx])
		h = torch.cat(h_concat, axis = -1)
		return h

	def forward(self, graph, feat, eweight = None):
		#graph = dgl.to_homogeneous(graph, ndata = ['x'])
		#feat = feat['molecule']
		feat = self.initial_layers(feat)
		for layer in self.gat_layers:
			feat = layer(graph, feat)
			feat = self.reshape_multihead_output(F.relu(feat))
		feat = self.out_layer(graph, feat)
		feat = torch.squeeze(feat, dim = 1)
		return feat

	def reset_parameters(self):
		for layer in self.gat_layers:
			layer.reset_parameters()