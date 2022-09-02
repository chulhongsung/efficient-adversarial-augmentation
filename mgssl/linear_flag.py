import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torch_geometric.data import DataLoader as G_DataLoader
from torch.utils.data import Dataset, DataLoader 

from mgssl.utils import MoleculeDataset, random_scaffold_split, Pretrained_feature_dataset
from mgssl.gnn_model import GNN_extractor
from mgssl.classifier import Linear_predictor

torch.cuda.is_available()
torch.cuda.get_device_name(0)
device = torch.device("cuda:0")

DATASET_NAME = "hiv"

dataset = MoleculeDataset(DATASET_NAME, dataset=DATASET_NAME)

smiles_list = pd.read_csv(DATASET_NAME + '/processed/smiles.csv', header=None)[0].tolist()
train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=0)

NUM_TASK = 1

### Train with Adversarial Augmentation

def train(model, loader, optimizer, step_size=0.001, m=3):
    model.train()

    for step, batch in enumerate(loader):
        batch_x, batch_y = batch
        
        perturb = torch.FloatTensor(batch_x.shape[0], batch_x.shape[1]).uniform_(-step_size, step_size).to(device)
        perturb.requires_grad_()

        pred = model(batch_x + perturb)

        y = batch_y.view(pred.shape).to(torch.float64)

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

            tmp_pred = model(batch_x + perturb)
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
    model = GNN_extractor(5, 300, JK='last', drop_ratio=0.0, graph_pooling = "mean", gnn_type='gin').to(device)
    model.from_pretrained("init.pth")
    model.eval()

    linear_model = Linear_predictor(300, 1).to(device)

    train_feature = []
    train_y = []
    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        with torch.no_grad():
            feature_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        train_feature.append(feature_)
        train_y.append(batch.y.view(len(feature_)))

    train_feature = torch.cat(train_feature, dim = 0)
    train_y = torch.cat(train_y, dim = 0)
    # train_y = torch.where(train_y == 1.0, train_y, 0.)

    val_feature = []
    val_y = []
    for step, batch in enumerate(val_loader):
        batch = batch.to(device)
        
        with torch.no_grad():
            feature_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        val_feature.append(feature_)
        val_y.append(batch.y.view(len(feature_)))

    val_feature = torch.cat(val_feature, dim = 0)
    val_y = torch.cat(val_y, dim = 0)
    # val_y = torch.where(val_y == 1.0, val_y, 0.)

    test_feature = []
    test_y = []
    for step, batch in enumerate(test_loader):
        batch = batch.to(device)
        
        with torch.no_grad():
            feature_ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        test_feature.append(feature_)
        test_y.append(batch.y.view(len(feature_)))

    test_feature = torch.cat(test_feature, dim = 0)
    test_y = torch.cat(test_y, dim = 0)
    #test_y = torch.where(test_y == 1.0, test_y, 0.)

    pre_train_dataset = Pretrained_feature_dataset(train_feature, train_y)
    pre_val_dataset = Pretrained_feature_dataset(val_feature, val_y)
    pre_test_dataset = Pretrained_feature_dataset(test_feature, test_y)
    
    pre_train_loader = DataLoader(pre_train_dataset, batch_size=100, shuffle=False)
    pre_val_loader = DataLoader(pre_val_dataset, batch_size=100, shuffle=False)
    pre_test_loader = DataLoader(pre_test_dataset, batch_size=100, shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss(reduction = "none")
    
    optimizer = optim.Adam(linear_model.parameters(), lr=0.01, weight_decay=0)
    print(optimizer)

    for epoch in range(1, 100):
        
        train(linear_model, pre_train_loader, optimizer)

        print("====Evaluation")

        print("Epoch:", epoch)
    
    print("Seed:", seed)