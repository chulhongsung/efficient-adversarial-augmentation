### GNN Extractor와 Linear classifier 동시에 진행

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

def train(model, loader, optimizer, step_size=0.001, max_pert=0.01, m=3):
    model.train()
    for _, batch in enumerate(loader):
        batch = batch.to(device)
        node_embedding = model.gnn(batch)
        perturb = torch.FloatTensor(node_embedding.shape[0], node_embedding.shape[1]).uniform_(-max_pert, max_pert).to(device)
        perturb.requires_grad_()
        graph_embedding = model.pool(node_embedding + perturb, batch.batch)
        pred = model.graph_pred_linear(graph_embedding)
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
            tmp_node_embedding =model.gnn(batch)
            tmp_graph_embedding = model.pool(tmp_node_embedding + perturb, batch.batch)
            tmp_pred = model.graph_pred_linear(tmp_graph_embedding)
            loss = 0
            loss_mat = criterion(tmp_pred.double(), (y+1)/2)
            loss += torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss /= m

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for seed in range(1):
    seed = 0
    train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

    train_loader = G_DataLoader(train_dataset, batch_size=500, shuffle=False, num_workers = 4)

    mgssl_bce_flag = GNN_graphpred(5, 300, num_tasks=1, JK='last', drop_ratio=0.2, graph_pooling='mean', gnn_type='gin').to(device)
    mgssl_bce_flag.from_pretrained("init.pth")

    criterion = nn.BCEWithLogitsLoss(reduction = "none")

    model_param_group = []
    model_param_group.append({"params": mgssl_bce_flag.gnn.parameters()})

    model_param_group.append({"params": mgssl_bce_flag.graph_pred_linear.parameters(), "lr":0.001*1})
    optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
    
    print(optimizer)

    eval_train = 0

    for epoch in range(1, 100):
        # print("====epoch " + str(epoch))
        
        train(mgssl_bce_flag, train_loader, optimizer, step_size=0.001, max_pert=0.01, m=3)

        print("Epoch:", epoch)

    print("Seed:", seed)

# torch.save(mgssl_bce_flag, "my_model.pth")

    