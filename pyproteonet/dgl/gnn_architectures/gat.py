import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GATConv
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .resettable_module import ResettableModule

class GAT(ResettableModule):
	def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
		super(GAT, self).__init__()
		self.layer1 = GATConv(in_feats = in_dim, out_feats = hidden_dim, num_heads=num_heads)
		# Be aware that the input dimension is hidden_dim*num_heads since
		# multiple head outputs are concatenated together. Also, only
		# one attention head in the output layer.
		hidden_dim_1 = hidden_dim//2
		self.layer2 = GATConv(in_feats = hidden_dim * num_heads, out_feats = hidden_dim_1, num_heads=num_heads)
		hidden_dim_2 = hidden_dim_1//2
		self.layer3 = GATConv(in_feats = hidden_dim_1 * num_heads, out_feats = out_dim, num_heads=1)
		
	def reshape_multihead_output(self, h):
		h_concat = []
		for h_idx in range(h.size()[1]):
			h_concat.append(h[:, h_idx])
		h = torch.cat(h_concat, axis = -1)
		return h

	def forward(self, graph, feat, eweight = None):
		graph = dgl.to_homogeneous(graph, ndata = ['x'])
		feat = feat['molecule']
		h, attention_1 = self.layer1(graph, feat, get_attention = True)
		h = self.reshape_multihead_output(F.relu(h))
		h, attention_2 = self.layer2(graph, h, get_attention = True)
		h = self.reshape_multihead_output(F.relu(h))
		h, attention_3 = self.layer3(graph, h, get_attention = True)
		h = torch.squeeze(h, dim = 2)
		return h

	def reset_parameters(self):
		self.layer1.reset_parameters()
		self.layer2.reset_parameters()
		self.layer3.reset_parameters()