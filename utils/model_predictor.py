import numpy as np
import torch.nn as nn
import torch
import dgl
from dgllife.model import HadamardLinkPredictor
import torch.nn.functional as F
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
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

        self.cnn_cell = nn.Sequential(nn.Conv1d(in_channels=1,
                                    out_channels=hidden_channels,
                                    kernel_size=kernel_size),
                                   nn.BatchNorm1d(hidden_channels),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv1d(in_channels=hidden_channels,
                                             out_channels=2,
                                             kernel_size=kernel_size),
                                   nn.BatchNorm1d(2),
                                   nn.LeakyReLU(inplace=True),
                                   )

        self.HA_predict = HadamardLinkPredictor(in_feats=2*predictor_hidden_feats,
                                      hidden_feats=graph_feat_size,
                                      num_layers=num_layers,
                                      n_tasks=1,activation=F.relu,
                                      dropout=predictor_dropout).to(device)

    def forward(self, bg, n_feats, e_feats,pathway,get_gradient=False):
        # print('bg: ',bg)
        # print(n_feats)
        node_feats = self.gnn_drug(bg, n_feats, e_feats)
        drug_feats = self.readout(bg, node_feats)
        #print('drug_feats: ',drug_feats.shape)
        pathway = pathway.requires_grad_()
        pathway_t = pathway.unsqueeze(2).permute(0, 2, 1)
        gene_feats=self.cnn_cell(pathway_t.float())
        # print(self.convs(pathway.unsqueeze(2)))
        pathway_feats = torch.nn.functional.adaptive_avg_pool1d(gene_feats,128)
        #print(pathway_feats.shape)
        pathway_feats=pathway_feats.view(pathway_feats.size()[0], -1)
        #print('pathway_feats: ',pathway_feats.shape)
        # Calculate graph representation by average readout.
        #Concat_hg = torch.mul(drug_feats, pathway_feats)
        #Concat_hg = torch.cat([graph_feats, pathway_feats], dim=1)
        Final_feature = self.HA_predict(drug_feats, pathway_feats)
        #print('Final_feature: ', Final_feature.shape)
        #print(Final_feature)
        if get_gradient:
            baseline1 = torch.zeros(node_feats.shape).to(device)
            scaled_nodefeats = [baseline1 + (float(i) / 50) * (node_feats - baseline1) for i in range(0, 51)]
            gradients=[]
            for scaled_nodefeat in scaled_nodefeats:
                scaled_hg = self.readout(bg, scaled_nodefeat)
                scaled_Final_feature = self.HA_predict(scaled_hg, pathway_feats)
                gradient = torch.autograd.grad(scaled_Final_feature[0][0], scaled_nodefeat)[0]
                gradient=gradient.detach().cpu().numpy()
                gradients.append(gradient)
            gradients=np.array(gradients)
            grads = (gradients[:-1] + gradients[1:]) / 2.0
            avg_grads = np.average(grads, axis=0)
            avg_grads=torch.from_numpy(avg_grads).to(device)
            # baseline=baseline.detach().cpu().numpy()
            # node_feats=node_feats.detach().cpu().numpy()
            integrated_gradients = (node_feats - baseline1) * avg_grads
            phi0 = []
            for j in range(node_feats.shape[0]):
                a = sum(integrated_gradients[j].detach().cpu().numpy().tolist())
                phi0.append(a)
            node_gradient = torch.tensor(phi0)

            baseline2 = torch.zeros(pathway.shape).to(device)
            scaled_pathways = [baseline2 + (float(i) / 50) * (pathway - baseline2) for i in range(0, 51)]
            gradients2 = []
            for scaled_pathway in scaled_pathways:
                # Calculate graph representation by av
                scaled_pathway_t = scaled_pathway.unsqueeze(2).permute(0, 2, 1)
                scaled_gene_feats = self.cnn_cell(scaled_pathway_t.float())
                # print(self.convs(pathway.unsqueeze(2)))
                scaled_pathway_feats = torch.nn.functional.adaptive_avg_pool1d(scaled_gene_feats, 128)
                # print(pathway_feats.shape)
                scaled_pathway_feats = scaled_pathway_feats.view(scaled_pathway_feats.size()[0], -1)
                scaled_Final_feature2 = self.HA_predict(drug_feats, scaled_pathway_feats)
                # target_label_idx = torch.argmax(scaled_Final_feature[0][3], 1).item()
                # index = np.ones((scaled_Final_feature.size()[0], 1)) * target_label_idx
                # index = torch.tensor(index, dtype=torch.int64).to(device)
                # output = scaled_Final_feature.gather(1, index)
                gradient2 = torch.autograd.grad(scaled_Final_feature2[0][0], scaled_pathway)[0]
                gradient2 = gradient2.detach().cpu().numpy()
                gradients2.append(gradient2)
            gradients2 = np.array(gradients2)
            # print(gradients.shape)
            grads2 = (gradients2[:-1] + gradients2[1:]) / 2.0
            avg_grads2 = np.average(grads2, axis=0)
            avg_grads2 = torch.from_numpy(avg_grads2).to(device)
            # baseline=baseline.detach().cpu().numpy()
            # node_feats=node_feats.detach().cpu().numpy()
            pathway_gradients = (pathway - baseline2) * avg_grads2

            return Final_feature,node_gradient,pathway_gradients
        else:
            return Final_feature

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
