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
from utils.focal_loss import FocalLoss
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
                                 task_names=['label'],
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
    total_auc = np.mean(train_meter.compute_metric('roc_auc_score'))
    total_prauc = np.mean(train_meter.compute_metric('pr_auc_score'))
    total_loss = np.mean(losses)
    if epoch % 10 == 0:
        print('epoch {:d}/{:d}, training_auc {:.4f}, training_prauc {:.4f}, training_loss {:.4f}'.format(epoch + 1, n_epochs, total_auc,total_prauc,
                                                                                  total_loss))
    return total_auc, total_prauc, total_loss

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
            val_losses.append(val_loss.data.item())
            eval_meter.update(vali_prediction, labels, masks)
        total_auc = np.mean(eval_meter.compute_metric('roc_auc_score'))
        total_prauc = np.mean(eval_meter.compute_metric('pr_auc_score'))
        total_loss = np.mean(val_losses)
    return total_auc, total_prauc, total_loss

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
        logit = model(bg, n_feats, e_feats, pathway)
        meter.update(logit, label, masks)
    auc = meter.compute_metric('roc_auc_score')
    auprc = meter.compute_metric('pr_auc_score')
    accuracy = meter.compute_metric('accuracy')
    f1 = meter.compute_metric('f1')
    return auc,auprc,accuracy,f1

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='gdsc',
                        help='gdsc or tcga')
    parser.add_argument('--epochs', type=int, default=501,
                        help='maximum number of epochs (default: 501)')
    parser.add_argument('--pretrain', type=str, default='True',
                        help='whether use pre-trained weights')
    parser.add_argument('--mode', type=str, default='test',
                        help='train or test')
    return parser.parse_args()

def main():
    args = arg_parse()
    if args.task == 'gdsc':
        path='../XGraphCDS/Data/class/'
        pathway_GDSC= pd.read_csv(path + 'GDSC_ssgesa.csv', index_col=0)
        pathway_col = pd.read_csv(path + 'pathway_name.csv', index_col=0).index.tolist()
        pathway_GDSC=pathway_GDSC[pathway_col].T
        pathresults='../XGraphCDS/Models/class/'
        for i in range(5):
            train_sets = pd.read_csv(path + str(i) + 'GDSC_label_train.csv')
            train_datasets = load_data(train_sets, pathway_GDSC, path + str(i) + 'GDSC_train', True, 'model_id')
            train_loader = DataLoader(train_datasets, batch_size=1024, shuffle=True,
                                      collate_fn=collate_molgraphs)

            valid_sets = pd.read_csv(path + str(i) + 'GDSC_label_valid.csv')
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
                    state_dict = torch.load(fn, map_location=torch.device('cpu'))
                    model.load_my_state_dict(state_dict)
                    model = model.to(device)
                elif args.pretrain == 'False':
                    model = model.to(device)

                n_epochs = args.epochs
                loss_fn = FocalLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-05)
                scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.00001, last_epoch=-1)

                min_score = 0.84
                for e in range(n_epochs):
                    scheduler.step()
                    run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
                    val_score = run_an_eval_epoch(model, valid_loader, loss_fn)
                    if e % 10 == 0:
                        print("第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))
                        print(
                            'epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, validation {} {:.4f}'.format(
                                e + 1, n_epochs, 'auc', val_score[0], 'prauc', val_score[1], 'loss', val_score[-1]))
                    if val_score[0] > min_score:
                        torch.save(model.state_dict(),
                                   os.path.join(pathresults, str(i) + 'model_{}.pth'.format(str(e))))

            elif args.mode == 'test':
                test_sets = pd.read_csv(path + 'GDSC_label_test.csv')
                test_datasets = load_data(test_sets, pathway_GDSC, path + 'GDSC_test', True, 'model_id')
                test_loader = DataLoader(test_datasets, batch_size=1024, shuffle=False,
                                         collate_fn=collate_molgraphs)

                PDX_sets = pd.read_csv(path + 'PDX_drug_labels.csv')
                pathway_PDX = pd.read_csv(path + 'PDX_ssgesa.csv', index_col=0)
                pathway_PDX = pathway_PDX[pathway_col].T
                PDX_datasets = load_data(PDX_sets, pathway_PDX, path + 'PDX', True, 'sample_id')
                PDX_loader = DataLoader(PDX_datasets, batch_size=128, shuffle=True,
                                        collate_fn=collate_molgraphs)

                TCGA_sets = pd.read_csv(path + 'tcga_drug_labels_test.csv')
                pathway_TCGA = pd.read_csv(path + 'tcga_ssgesa.csv', index_col=0)
                pathway_TCGA = pathway_TCGA[pathway_col].T
                TCGA_datasets = load_data(TCGA_sets, pathway_TCGA, path + 'TCGA', True, 'sample_id')
                TCGA_loader = DataLoader(TCGA_datasets, batch_size=128, shuffle=True,
                                             collate_fn=collate_molgraphs)

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
                # 加载模型
                fn = '../XGraphCDS/Models/class/' + str(i) + 'model.pth'
                model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
                gcn_net = model.to(device)


                score = eval(test_loader, gcn_net)
                print('Fold'+str(i)+': '+
                    'test {} {:.4f}, test {} {:.4f}, test {} {:.4f}, test {} {:.4f}'.format(
                        'auc', score[0][0], 'prauc', score[1][0],
                        'accuracy', score[2][0], 'f1', score[-1][0]))
                score = eval(PDX_loader, gcn_net)
                print('Fold'+str(i)+': '+
                    'PDX {} {:.4f}, PDX {} {:.4f}, PDX {} {:.4f}, PDX {} {:.4f}'.format(
                       'auc', score[0][0], 'prauc', score[1][0],
                        'accuracy', score[2][0], 'f1', score[-1][0]))
                score = eval(TCGA_loader, gcn_net)
                print('Fold'+str(i)+': '+
                    'TCGA {} {:.4f}, TCGA {} {:.4f}, TCGA {} {:.4f}, TCGA {} {:.4f}'.format(
                        'auc', score[0][0], 'prauc', score[1][0],
                        'accuracy', score[2][0], 'f1', score[-1][0]))

    elif args.task == 'tcga':
        path='../XGraphCDS/Data/tcga/'
        pathway_TCGA = pd.read_csv(path + 'tcga_ssgesa.csv', index_col=0)
        pathway_col = pd.read_csv(path + 'pathway_name.csv', index_col=0).index.tolist()
        pathway_TCGA = pathway_TCGA[pathway_col].T
        pathresults='../XGraphCDS/Models/tcga/'
        for i in range(5):
            train_sets = pd.read_csv(path + str(i) + 'tcga_label_train.csv')
            train_datasets = load_data(train_sets, pathway_TCGA, path + str(i) + 'tcga_train', True, 'patient_id')
            train_loader = DataLoader(train_datasets, batch_size=256, shuffle=True,
                                      collate_fn=collate_molgraphs)
            valid_sets = pd.read_csv(path + str(i) + 'tcga_label_valid.csv')
            valid_datasets = load_data(valid_sets, pathway_TCGA, path + str(i) + 'tcga_valid', True, 'patient_id')
            valid_loader = DataLoader(valid_datasets, batch_size=128, shuffle=True,
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
                    fn = '../XGraphCDS/Models/model.pth'
                    model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
                    model = model.to(device)
                elif args.pretrain == 'False':
                    model = model.to(device)

                n_epochs = args.epochs
                loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-05)
                scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.0035, last_epoch=-1)

                min_score = 0.81
                for e in range(n_epochs):
                    scheduler.step()
                    run_a_train_epoch(n_epochs, e, model, train_loader, loss_fn, optimizer)
                    val_score = run_an_eval_epoch(model, valid_loader, loss_fn)
                    if e % 10 == 0:
                        print("第%d个epoch的学习率：%f" % (e + 1, optimizer.param_groups[0]['lr']))
                        print(
                            'epoch {:d}/{:d}, validation {} {:.4f}, validation {} {:.4f}, validation {} {:.4f}'.format(
                                e + 1, n_epochs, 'auc', val_score[0], 'prauc', val_score[1], 'loss', val_score[-1]))
                    if val_score[0] > min_score:
                        torch.save(model.state_dict(),
                                   os.path.join(pathresults, str(i) + 'model_{}.pth'.format(str(e))))

            elif args.mode == 'test':
                test_sets = pd.read_csv(path + 'tcga_drug_labels_ssgesa_test.csv')
                test_datasets = load_data(test_sets, pathway_TCGA, path + 'tcga_test', True, 'patient_id')
                test_loader = DataLoader(test_datasets, batch_size=128, shuffle=False,
                                         collate_fn=collate_molgraphs)

                PDX_sets = pd.read_csv(path + 'PDX_drug_labels.csv')
                pathway_PDX = pd.read_csv(path + 'PDX_ssgesa.csv', index_col=0)
                pathway_PDX = pathway_PDX[pathway_col].T
                PDX_datasets = load_data(PDX_sets, pathway_PDX, path + 'PDX', True, 'sample_id')
                PDX_loader = DataLoader(PDX_datasets, batch_size=128, shuffle=True,
                                        collate_fn=collate_molgraphs)


                model = ModelPredictor(node_feat_size=atom_featurizer.feat_size('hv'),
                                       edge_feat_size=bond_featurizer.feat_size('he'),
                                       num_layers=2,
                                       num_timesteps=2,
                                       predictor_hidden_feats=128,
                                       graph_feat_size=256,
                                       #dropout=0.3,
                                       #predictor_dropout=0.3,
                                       n_tasks=1,
                                       kernel_size=5,
                                       hidden_channels=10)
                # 加载模型
                fn = '../XGraphCDS/Models/tcga/' + str(i) + 'model.pth'
                model.load_state_dict(torch.load(fn, map_location=torch.device('cpu')))
                gcn_net = model.to(device)


                score = eval(test_loader, gcn_net)
                print('Fold'+str(i)+': '+
                    'test {} {:.4f}, test {} {:.4f}, test {} {:.4f}, test {} {:.4f}'.format(
                        'auc', score[0][0], 'prauc', score[1][0],
                        'accuracy', score[2][0], 'f1', score[-1][0]))
                score = eval(PDX_loader, gcn_net)
                print('Fold'+str(i)+': '+
                    'PDX {} {:.4f}, PDX {} {:.4f}, PDX {} {:.4f}, PDX {} {:.4f}'.format(
                        'auc', score[0][0], 'prauc', score[1][0],
                        'accuracy', score[2][0], 'f1', score[-1][0]))


if __name__ == "__main__":
    main()

