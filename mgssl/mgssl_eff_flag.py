import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from sklearn.metrics import  precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve

import pandas as pd

import os
import shutil

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch_geometric.data import DataLoader as G_DataLoader

from mgssl.utils import MoleculeDataset, random_scaffold_split
from mgssl.gnn_model import GNN_graphpred
from utils.influence import compute_loo_if

torch.cuda.is_available()
torch.cuda.get_device_name(0)
device = torch.device("cuda:0")

DATASET_NAME = "hiv"

dataset = MoleculeDataset(DATASET_NAME, dataset=DATASET_NAME)

smiles_list = pd.read_csv(DATASET_NAME + '/processed/smiles.csv', header=None)[0].tolist()
train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=0)

NUM_TASK = 1

def train(model, loader, optimizer, k=50, step_size=0.001, max_pert=0.01, m=3):
    model.train()

    for _, batch in enumerate(loader):
        batch = batch.to(device)
        embedding = model.pool(model.gnn(batch), batch.batch)
        loo_influence = compute_loo_if(model.graph_pred_linear, embedding, batch.y, criterion)
        _, topk_idx = torch.topk(loo_influence, k=k, axis=-1)
        
        perturb = torch.FloatTensor(k, 300).uniform_(-max_pert, max_pert).to(device)
        perturb.requires_grad_()
        
        embedding[topk_idx, :] = embedding[topk_idx, :] + perturb
        pred = model.graph_pred_linear(embedding)
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
            perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
            perturb.data = perturb_data.data
            perturb.grad[:] = 0
            tmp_embedding = model.pool(model.gnn(batch), batch.batch)
            tmp_embedding[topk_idx, :] = tmp_embedding[topk_idx, :] + perturb
            tmp_pred = model.graph_pred_linear(tmp_embedding)
            loss = 0
            loss_mat = criterion(tmp_pred.double(), (y+1)/2)
            loss += torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss /= m

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

test_auc_list = []

for seed in range(1):
    seed = 0
    train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

    train_loader = G_DataLoader(train_dataset, batch_size=500, shuffle=False, num_workers = 4)
    val_loader = G_DataLoader(valid_dataset, batch_size=500, shuffle=False, num_workers = 4)
    test_loader = G_DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers = 4)

    mgssl_bce_eff_flag = GNN_graphpred(5, 300, num_tasks=1, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)
    mgssl_bce_eff_flag.from_pretrained("init.pth")

    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    model_param_group = []
    model_param_group.append({"params": mgssl_bce_eff_flag.gnn.parameters()})

    model_param_group.append({"params": mgssl_bce_eff_flag.graph_pred_linear.parameters(), "lr":0.001*1})
    optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
    
    print(optimizer)

    eval_train = 0

    for epoch in range(1, 100):
        
        train(mgssl_bce_eff_flag, train_loader, optimizer)

        print("Epoch:", epoch)
    
    print("Seed:", seed)
    # torch.save(linear_model, "/content/gdrive/MyDrive/MGSSL/linear_epoch700_0729_hiv_par_2_07_" + str(seed) + ".pth")

    