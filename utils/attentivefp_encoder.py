import torch.nn as nn
import torch
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'

class ModelPredictor(nn.Module):
    def __init__(self, node_feat_size,edge_feat_size, graph_feat_size, activation=None, aggregator_type=None,num_layers=None,
                 dropout=0., classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1,num_timesteps=None,
                 predictor_hidden_feats=128, predictor_dropout=0.,hidden_channels=None,
                 kernel_size=None):
        super(ModelPredictor, self).__init__()

        self.gnn_drug = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)

        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        # self.predict = nn.Sequential(nn.Dropout(dropout),
        #                              nn.Linear(hidden_feats, n_tasks),
        #                              nn.Sigmoid())

    def forward(self, bg, n_feats, e_feats):
        node_feats = self.gnn_drug(bg, n_feats, e_feats)
        drug_feats = self.readout(bg, node_feats)
        return drug_feats #,self.predict(drug_feats)