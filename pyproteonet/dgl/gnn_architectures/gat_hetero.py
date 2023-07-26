import dgl
import numpy as np
import os
import torch
import torch.nn as nn
from dgl.nn import GATConv
from dgl.nn.pytorch import HeteroGraphConv
import torch.nn.functional as F
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
from dgl.nn import AvgPooling, GNNExplainer

class GAT_hetero(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
		super(GAT_hetero, self).__init__()
		self.layer1 = HeteroGraphConv({
			'interacts': GATConv(in_feats = in_dim, out_feats = hidden_dim, num_heads=num_heads), 
			'external': GATConv(in_feats = in_dim, out_feats = hidden_dim, num_heads=num_heads)
			}, aggregate='sum') 
		# Be aware that the input dimension is hidden_dim*num_heads since
		# multiple head outputs are concatenated together. Also, only
		# one attention head in the output layer.
		hidden_dim_1 = hidden_dim//2
		self.layer2 = HeteroGraphConv({
			'interacts': GATConv(in_feats = hidden_dim * num_heads, out_feats = hidden_dim_1, num_heads=num_heads), 
			'external': GATConv(in_feats = hidden_dim * num_heads, out_feats = hidden_dim_1, num_heads=num_heads)
			}, aggregate='sum')

		hidden_dim_2 = hidden_dim_1//2
		self.layer3 = HeteroGraphConv({
			'interacts': GATConv(in_feats = hidden_dim_1 * num_heads, out_feats = hidden_dim_2, num_heads=num_heads), 
			'external': GATConv(in_feats = hidden_dim_1 * num_heads, out_feats = hidden_dim_2, num_heads=num_heads)
			}, aggregate='sum') 
		
		hidden_dim_3 = hidden_dim_2//2
		self.layer4 = HeteroGraphConv({
			'interacts': GATConv(in_feats = hidden_dim_2 * num_heads, out_feats = out_dim, num_heads=1), 
			'external': GATConv(in_feats = hidden_dim_2 * num_heads, out_feats = out_dim, num_heads=1)
			}, aggregate='sum') 

	def reshape_multihead_output(self, h):
		h_concat = []
		for h_idx in range(h.size()[1]):
			h_concat.append(h[:, h_idx])
		h = torch.cat(h_concat, axis = -1)
		h = {'molecule': h}
		return h

	def forward(self, graph, feat, eweight = None):
		h = self.layer1(graph, feat)
		h = self.reshape_multihead_output(F.relu(h['molecule']))
		h = self.layer2(graph, h)
		h = self.reshape_multihead_output(F.relu(h['molecule']))
		h = self.layer3(graph, h)
		h = self.reshape_multihead_output(F.relu(h['molecule']))
		h = self.layer4(graph, h)
		h = torch.squeeze(h['molecule'], dim = 2)
		return h