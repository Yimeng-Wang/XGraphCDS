# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Creating datasets from .csv files for molecular property prediction.

import dgl.backend as F
import numpy as np
import os
import torch

from dgl.data.utils import save_graphs, load_graphs

from dgllife.utils.io import pmap

__all__ = ['MoleculeCSVDataset']



class MoleculeCSVDataset(object):

    def __init__(self, df, df2,smiles_to_graph, node_featurizer, edge_featurizer, smiles_column,
                 model_column, cache_file_path, task_names=None, load=False, log_every=1000,
                 init_mask=True,n_jobs=1):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        #print(np.array(self.smiles).shape)
        self.model_id = self.df[model_column].tolist()
        self.df2 = df2
        self.features = np.array(self.df2[self.model_id]).T
        #print(self.features.shape)
        self.features = torch.tensor(self.features)
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer,
                          load, log_every, init_mask, n_jobs)

        # Only useful for binary classification tasks
        self._task_pos_weights = None

    def _pre_process(self, smiles_to_graph, node_featurizer,
                     edge_featurizer, load, log_every, init_mask, n_jobs=1):

        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(smiles_to_graph,
                                   self.smiles,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i+1, len(self)))
                    self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                       edge_featurizer=edge_featurizer))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):

        if self.mask is not None:
            #self.features[item] = torch.tensor(self.features[item])
            return self.smiles[item], self.graphs[item], self.model_id[item],self.features[item],self.labels[item], self.mask[item]
        else:
            #self.features[item] = torch.tensor(self.features[item])
            return self.smiles[item], self.graphs[item], self.model_id[item],self.features[item],self.labels[item]

    def __len__(self):

        return len(self.smiles)

    def task_pos_weights(self, indices):

        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights
