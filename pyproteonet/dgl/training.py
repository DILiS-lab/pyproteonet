from typing import Optional
import os

import numpy as np
import torch
import psutil
import torch.nn.functional as F

from .graph_data_set import GraphDataSet


def train_node_regression(model: torch.nn.Module, train_data_set: GraphDataSet, test_data_set: Optional[GraphDataSet] = None,
                          nan_substitute_value: float = 0.0, mask_substitute_value  : float = 0.0,
                          num_epochs: int = 1000, print_frequency: int = -1, test_frequency: int = 1,
                          hide_substitute_value: float = 0.0, random_seed: Optional[int] = None,
                          device:str = 'cpu'):
    rng = np.random.default_rng()
    if print_frequency < 0:
        print_frequency = len(train_data_set)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    iteration = 0
    for e in range(num_epochs):
        train_loss_values, train_MAEs, train_r2s = [], [], []
        model.train()
        train_indices = np.arange(len(train_data_set))
        rng.shuffle(train_indices)
        for i, graph_id in enumerate(train_indices):
            graph = train_data_set[graph_id]
            features = graph.nodes['molecule'].data['x'].float()
            #molecule_type = graph.nodes['molecule'].data['type'].numpy()
            abundances = graph.nodes['molecule'].data['abundance']
            train_nodes = graph.ndata['mask']
            features_train = features.clone()
            if 'hide' in graph.ndata:
                to_hide = graph.ndata['hide']
                features_train[to_hide, 0] = hide_substitute_value
            features_train[train_nodes, 0] = mask_substitute_value
            features_train[features_train.isnan()] = nan_substitute_value
            features_dict = {'molecule': features_train}
            #print(features_train)
            # Forward
            pred = model(graph, feat = features_dict)
            # Loss
            abundances_train = abundances[train_nodes]
            abundances_train[abundances_train.isnan()] = nan_substitute_value
            train_loss = F.mse_loss(pred[train_nodes], abundances_train)
            train_loss_values.append(train_loss.detach().numpy())
            train_MAEs.append(F.l1_loss(pred[train_nodes], abundances_train).detach().numpy())
            train_r2s.append((torch.corrcoef(torch.t(torch.cat((pred[train_nodes], abundances_train), 1)))[0, 1]**2).detach().numpy())
            # Backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if print_frequency > 0 and (iteration+1) % print_frequency == 0:
                loss = np.array(train_loss_values).mean()
                mae = np.array(train_MAEs).mean()
                r2 = np.array(train_r2s).mean()
                memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                print(f"Epoch {e+1} iteration {i+1}: train_loss:{loss}, train_MAE:{mae}, train_R2:{r2}, memory:{memory}")
            iteration+=1
        if test_data_set is not None and (e+1)%test_frequency == 0:
            model.eval()
            with torch.no_grad():
                test_loss_values, test_MAEs, test_r2s = [], [], []
                for i, graph in enumerate(test_data_set):
                    features = graph.nodes['molecule'].data['x'].float()
                    abundances = graph.nodes['molecule'].data['abundance']
                    mask_nodes = graph.nodes['molecule'].data['mask']
                    features_test = features.clone()
                    if 'hide' in graph.nodes['molecule'].data:
                        to_hide = graph.nodes['molecule'].data['hide']
                        features_test[to_hide, 0] = hide_substitute_value
                    features_test[mask_nodes, 0] = mask_substitute_value
                    features_test[features_test.isnan()] = nan_substitute_value
                    features_dict = {'molecule': features_test}
                    pred = model(graph, feat = features_dict)
                    abundances_test = abundances[mask_nodes]
                    abundances_test[abundances_test.isnan()] = nan_substitute_value
                    test_loss = F.mse_loss(pred[mask_nodes], abundances_test)
                    test_loss_values.append(test_loss.detach().numpy())
                    test_MAEs.append(F.l1_loss(pred[mask_nodes],abundances_test).detach().numpy())
                    test_r2s.append((torch.corrcoef(torch.t(torch.cat((pred[mask_nodes], abundances_test), 1)))[0, 1]**2).detach().numpy())
                loss = np.array(test_loss_values).mean()
                mae = np.array(test_MAEs).mean()
                r2 = np.array(test_r2s).mean()
                memory = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
                print(f"Epoch {e+1}: test_loss:{loss}, test_MAE:{mae}, test_R2:{r2}, memory:{memory}")