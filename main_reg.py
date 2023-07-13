import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader

from dgllife.utils import smiles_to_bigraph
from utils.eval_meter import Meter
import torch.nn as nn
from utils.model_predictor import ModelPredictor
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.featurizers import CanonicalAtomFeaturizer,CanonicalBondFeaturizer

if torch.cuda.is_available():
    print('use GPU')
    device = 'cuda'
else:
    print('use CPU')
    device = 'cpu'


from functools import partial

import argparse
import os
import random
import numpy as np
seed = 16
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)

def set_random_seed(seed=16):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed = 16
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_molgraphs(data):
    assert len(data[0]) in [5, 6], \
        'Expect the tuple to be of length 4 or 5, got {:d}'.format(len(data[0]))
    if len(data[0]) == 5:
        smiles, graphs,model_id, pathway, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, model_id, pathway, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    pathway = torch.stack(pathway, dim=0)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles,  bg, model_id, pathway, labels, masks

from utils.csv_dataset import MoleculeCSVDataset
atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_featurizer = CanonicalBondFeaturizer(bond_data_field='he', self_loop=True)
def load_data(df1,df2,path,load,model_id):
    dataset = MoleculeCSVDataset(df1,df2,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop='self_loop'),
                                 node_featurizer=atom_featurizer,
                                 edge_featurizer=bond_featurizer,
                                 smiles_column='SMILES',
                                 model_column=model_id,
                                 cache_file_path=path+ '_graph.bin',
                                 task_names=['IC50'],
                                 load=load,init_mask=True,n_jobs=-1
                                 )

    return dataset

def run_a_train_epoch(n_epochs,epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    losses = []
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, model_id, pathway,labels, masks = batch_data
        bg=bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        pathway = pathway.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        prediction = model(bg, n_feats, e_feats, pathway)
        loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(prediction, labels, masks)
        losses.append(loss.data.item())
    total_r2 = np.mean(train_meter.compute_metric('r2'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training r2 {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_r2 ,total_loss))
    return total_r2, total_loss

def run_an_eval_epoch(model, data_loader,loss_criterion):
    model.eval()
    val_losses=[]
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg,model_id,  pathway,labels, masks = batch_data
            bg = bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            pathway = pathway.to(device)
            n_feats = bg.ndata.pop('hv').to(device)
            e_feats = bg.edata.pop('he').to(device)
            vali_prediction = model(bg, n_feats, e_feats, pathway)
            val_loss = (loss_criterion(vali_prediction, labels) * (masks != 0).float()).mean()
            val_loss=val_loss.detach().cpu().numpy()
            val_losses.append(val_loss)
            eval_meter.update(vali_prediction, labels, masks)
        total_score = np.mean(eval_meter.compute_metric('r2'))
        total_loss = np.mean(val_losses)
    return total_score, total_loss

def eval(dataloader, model):
    model.eval()
    meter = Meter()
    for batch_id, batch_data in enumerate(dataloader):
        smiles, bg, model_id,  pathway,label, masks = batch_data
        bg = bg.to(device)
        pathway=pathway.to(device)
        label = label.to(device)
        masks = masks.to(device)
        n_feats = bg.ndata.pop('hv').to(device)
        e_feats = bg.edata.pop('he').to(device)
        pred = model(bg, n_feats,e_feats, pathway)
        meter.update(pred, label, masks)
    PCC = meter.compute_metric('pcc')
    R2 = meter.compute_metric('r2')
    MAE = meter.compute_metric('mae')
    RMSE = meter.compute_metric('rmse')
    return PCC, R2, MAE, RMSE

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1001,
                        help='maximum number of epochs (default: 1001)')
    parser.add_argument('--pretrain', type=str, default='True',
                        help='whether use pre-trained weights')
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test')
    return parser.parse_args()


def main():
    args = arg_parse()
    path='../XGraphCDS/Data/reg/'
    pathway_GDSC= pd.read_csv(path + 'GDSC_ssgesa.csv', index_col=0)
    pathway_col = pd.read_csv(path + 'pathway_name.csv', index_col=0).index.tolist()
    pathway_GDSC=pathway_GDSC[pathway_col].T
    pathresults='../XGraphCDS/Models/reg/'
    for i in range(5):
        train_sets = pd.read_csv(path + str(i) + 'GDSC_IC50s_train.csv')
        train_datasets = load_data(train_sets, pathway_GDSC, path + str(i) + 'GDSC_train', True, 'model_id')
        train_loader = DataLoader(train_datasets, batch_size=1024, shuffle=True,
                                  collate_fn=collate_molgraphs)

        valid_sets = pd.read_csv(path + str(i) + 'GDSC_IC50s_valid.csv')
        valid_datasets = load_data(valid_sets, pathway_GDSC, path + str(i)+'GDSC_valid', True, 'model_id')
        valid_loader = DataLoader(valid_datasets, batch_size=1024, shuffle=True,
                                  collate_fn=collate_molgraphs)
        if args.mode == 'train':
            model = ModelPredictor(node_feat_size=atom_featurizer.feat_size('hv'),
                                   edge_feat_size=bond_featurizer.feat_size('he'),
                                   num_layers=2,
                                   num_timesteps=2,
                                   predictor_hidden_feats=128,
                                   graph_feat_size=256,
                                   dropout=0.3,
                                   predictor_dropout=0.3,
                                   n_tasks=1,
                                   kernel_size=5,
                                   hidden_channels=10)
            if args.pretrain == 'True':
                fn = '../XGraphCDS/Models/model_80.pth'
                state_dict = torch.load(fn,map_location=torch.device('cpu'))
                model.load_my_state_dict(state_dict)
                model = model.to(device)
            elif args.pretrain == 'False':
                model = model.to(device)

            n_epochs = args.epochs
            loss_fn = nn.MSELoss(reduction='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-05)
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.00005, last_epoch=-1)

            min_score = 0.85
            for e in range(n_epochs):
                scheduler.step()
                run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
                val_score = run_an_eval_epoch(model, valid_loader, loss_fn)
                if e % 10 == 0:
                    print("第%d个epoch的学习率：%f" % (e+1, optimizer.param_groups[0]['lr']))
                    print('epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}'.format(
                        e + 1, n_epochs, 'r2', val_score[0], 'loss', val_score[-1]))
                if val_score[0] > min_score:
                    torch.save(model.state_dict(), os.path.join(pathresults, str(i)+'model_{}.pth'.format(str(e))))

        elif args.mode == 'test':
            test_sets = pd.read_csv(path + 'GDSC_IC50s_test.csv')
            test_datasets = load_data(test_sets, pathway_GDSC, path + 'GDSC_test', True, 'model_id')
            test_loader = DataLoader(test_datasets, batch_size=1024, shuffle=False,
                                     collate_fn=collate_molgraphs)

            test_sets1 = pd.read_csv(path + 'GDSC_IC50s_olddrugoldcell.csv')
            test_datasets1 = load_data(test_sets1, pathway_GDSC, path + 'GDSC_olddrugoldcell', True, 'model_id')
            test_loader1 = DataLoader(test_datasets1, batch_size=128, shuffle=False,
                                      collate_fn=collate_molgraphs)

            test_sets2 = pd.read_csv(path + 'GDSC_IC50s_newdrugnewcell.csv')
            test_datasets2 = load_data(test_sets2, pathway_GDSC, path + 'GDSC_newdrugnewcell', True, 'model_id')
            test_loader2 = DataLoader(test_datasets2, batch_size=128, shuffle=True,
                                      collate_fn=collate_molgraphs)

            model = ModelPredictor(node_feat_size=atom_featurizer.feat_size('hv'),
                                   edge_feat_size=bond_featurizer.feat_size('he'),
                                   num_layers=2,
                                   num_timesteps=2,
                                   predictor_hidden_feats=128,
                                   graph_feat_size=256,
                                   # dropout=0.1,
                                   # predictor_dropout=0.1,
                                   n_tasks=1,
                                   kernel_size=5,
                                   hidden_channels=10)
            # 加载模型
            fn = '../XGraphCDS/Models/reg/' + str(i) + 'model.pth'
            model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
            gcn_net = model.to(device)

            score = eval(test_loader, gcn_net)
            #print(score[0][0])
            print('Fold'+str(i)+': '+
                'test {} {:.4f}, test {} {:.4f}, test {} {:.4f}, test {} {:.4f}'.format(
                'pcc', score[0][0], 'r2', score[1][0],
                'mae', score[2][0], 'rmse', score[-1][0]))
            score = eval(test_loader1, gcn_net)
            print('Fold'+str(i)+': '+
                'olddrugoldcell {} {:.4f}, olddrugoldcell {} {:.4f}, olddrugoldcell {} {:.4f}, olddrugoldcell {} {:.4f}'.format(
                'pcc', score[0][0], 'r2', score[1][0],
                'mae', score[2][0], 'rmse', score[-1][0]))
            score = eval(test_loader2, gcn_net)
            print('Fold'+str(i)+': '+
                'newdrugnewcell {} {:.4f}, newdrugnewcell {} {:.4f}, newdrugnewcell {} {:.4f}, newdrugnewcell {} {:.4f}'.format(
                'pcc', score[0][0], 'r2', score[1][0],
                'mae', score[2][0], 'rmse', score[-1][0]))

if __name__ == "__main__":
    main()
