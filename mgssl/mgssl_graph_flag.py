import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import pandas as pd

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from mgssl.utils import MoleculeDataset, random_scaffold_split
from torch_geometric.data import DataLoader as G_DataLoader
from mgssl.gnn_model import GNN_graphpred

torch.cuda.is_available()
torch.cuda.get_device_name(0)
device = torch.device("cuda:0")

DATASET_NAME = "hiv"

dataset = MoleculeDataset(DATASET_NAME, dataset=DATASET_NAME)

smiles_list = pd.read_csv(DATASET_NAME + '/processed/smiles.csv', header=None)[0].tolist()
train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=0)

NUM_TASK = 1

def train(model, loader, optimizer, step_size=0.001, m=3):
    model.train()
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        perturb = torch.FloatTensor(batch.id.shape[0], 300).uniform_(-0.01, 0.01).to(device)
        perturb.requires_grad_()
        embedding = model.pool(model.gnn(batch), batch.batch)
        pred = model.graph_pred_linear(embedding + perturb)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss /= m
        for _ in range(m - 1):
            loss.backward()
            perturb_data = perturb.detach() + 0.001 * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            tmp_embedding = model.pool(model.gnn(batch), batch.batch)
            tmp_pred = model.graph_pred_linear(tmp_embedding + perturb)
            loss = 0
            loss_mat = criterion(tmp_pred.double(), (y+1)/2)
            loss += torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss /= m

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# def eval(model, loader):
#     model.eval()
#     y_true = []
#     y_scores = []

#     for step, batch in enumerate(loader):
#         batch_x, batch_y = batch

#         with torch.no_grad():
#             pred = model(batch_x)

#         y_true.append(batch_y.view(pred.shape))
#         y_scores.append(pred)

#     y_true = torch.cat(y_true, dim = 0).cpu().numpy()
#     y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    
#     roc_list = []
    
#     for i in range(y_true.shape[1]):
#         #AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
#             is_valid = y_true[:,i]**2 > 0
#             roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
#             # acc_list.append(np.mean((y_true[is_valid, i] +1)/2 == y_pred[is_valid,i]))

#     if len(roc_list) < y_true.shape[1]:
#         print("Some target is missing!")
#         print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

#     return sum(roc_list)/len(roc_list)

test_auc_list = []

for seed in range(1):
    seed = 0
    train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

    train_loader = G_DataLoader(train_dataset, batch_size=500, shuffle=False, num_workers = 4)
    val_loader = G_DataLoader(valid_dataset, batch_size=500, shuffle=False, num_workers = 4)
    test_loader = G_DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 4)

    # from tensorboardX import SummaryWriter

    mgssl_bce = GNN_graphpred(5, 300, num_tasks=1, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)
    mgssl_bce.from_pretrained("init.pth")

    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    model_param_group = []
    model_param_group.append({"params": mgssl_bce.gnn.parameters()})

    model_param_group.append({"params": mgssl_bce.graph_pred_linear.parameters(), "lr":0.001*1})
    optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
    
    print(optimizer)

    eval_train = 0

    for epoch in range(1, 100):
        # print("====epoch " + str(epoch))
        
        train(mgssl_bce, train_loader, optimizer)

        # print("====Evaluation")
        # if eval_train:
        #     train_auc, _ = eval(mgssl_bce, train_loader, optimizer)
        # else:
        #     # print("omit the training accuracy computation")""
        #     train_acc = 0
        # val_auc, _ = eval(linear_model, pre_val_loader)
        # test_auc, _ = eval(linear_model, pre_test_loader)
        print("Epoch:", epoch)
        # print("AUC train: %f val: %f test: %f" %(train_auc, val_auc, test_auc))
        
# torch.save(mgssl_par,'/content/gdrive/MyDrive/MGSSL/mgssl_hiv_par_2_08.pth')
    
    print("Seed:", seed)
    # test_auc_list.append(test_auc)
    # torch.save(linear_model, "/content/gdrive/MyDrive/MGSSL/linear_epoch700_0729_hiv_par_2_07_" + str(seed) + ".pth")

    