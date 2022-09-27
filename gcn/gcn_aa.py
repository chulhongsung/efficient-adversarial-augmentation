import sys
sys.path.append('../')

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data, DataLoader, InMemoryDataset
# from torch.utils.data import Dataset, DataLoader 

from model import GNN_graphpred
from module import MoleculeDataset, random_scaffold_split

from sklearn.metrics import(
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score
)

torch.cuda.is_available()
torch.cuda.get_device_name(0)
device = torch.device("cuda:0")

DATASET_NAME = "hiv"
NUM_TASK = 1

dataset = MoleculeDataset('../dataset/' + DATASET_NAME, dataset=DATASET_NAME)

smiles_list = pd.read_csv('../dataset/' + DATASET_NAME + '/processed/smiles.csv', header=None)[0].tolist()
# train_dataset, valid_dataset, test_dataset, train_scaffold_idx = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=0)


### Train with Adversarial Augmentation
def sigmoid(z):
    return 1/(1 + np.exp(-z))


def train(model, loader, optimizer, step_size=0.001, max_pert=0.01, m=3):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        perturb = torch.FloatTensor(batch.id.shape[0], 300).uniform_(-max_pert, max_pert).to(device)
        perturb.requires_grad_()
        
        graph_embedding = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        pred = model.graph_pred_linear(graph_embedding + perturb)

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
            
            tmp_graph_embedding = model.pool(model.gnn(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            tmp_pred = model.graph_pred_linear(tmp_graph_embedding + perturb)

            loss = 0
            loss_mat = criterion(tmp_pred.double(), (y+1)/2)
            loss += torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss /= m

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()


@torch.no_grad()
def eval(model, device, loader):
    model.eval()
    
    y_true = []
    y_score = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_score.append(pred)
    
    y_true = torch.cat(y_true, dim = 0).data.cpu().numpy()
    y_score = torch.cat(y_score, dim = 0).data.cpu().numpy()
    y_pred = np.where(sigmoid(y_score) > 0.5, 1.0, 0.0)
    
    roc_list = []
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_score[is_valid,i]))
            #acc_list.append(np.mean((y_true[is_valid, i] +1)/2 == y_pred[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), roc_list, y_pred 



if __name__ == '__main__':
    test_roc_list = []
    test_prec_list = []
    test_recall_list = []
    test_f1_list = []
    test_acc_list = []

    for seed in range(10):
        # seed = 0
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed=seed)

        train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=False, num_workers = 1)
        val_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers = 1)
        test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = 1)
        
        model = GNN_graphpred(5, 300, num_tasks=NUM_TASK, JK='last', drop_ratio=0.2, graph_pooling='mean').to(device)
        
        criterion = nn.BCEWithLogitsLoss(reduction = "none")

        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})

        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":0.001*1})
        optimizer = optim.Adam(model_param_group, lr=0.001, weight_decay=0)
        print(optimizer)
        
        eval_train = 1
        test_roc_per_epochs = []

        for epoch in range(1, 30+1):
            # print("====epoch " + str(epoch))
            
            train(model, train_loader, optimizer, step_size=0.001, max_pert = 0.01, m=3)

            # print("====Evaluation")
            if eval_train:
                train_roc, train_roc_list, train_pred = eval(model, device, train_loader)
            else:
                # print("omit the training accuracy computation")
                train_roc = 0
            val_roc, val_roc_list, val_pred = eval(model, device, val_loader)
            test_roc, test_roc_list_, test_pred = eval(model, device, test_loader)
            
            test_roc_per_epochs.append(test_roc)

            print("epoch: %f train: %f val: %f test: %f" %(epoch, train_roc, val_roc, test_roc))

        print("Seed:", seed)
        
        test_y = []
        for d, s in enumerate(test_dataset):
            y_tmp = [0 if i == -1 else i for i in s.y.numpy()]
            test_y.append(y_tmp[0])
            
        pred = [int(i[0]) for i in test_pred]
        
        test_roc_list.append(max(test_roc_per_epochs))
        # test_acc_list.append(accuracy_score(test_y, pred))
        # test_prec_list.append(precision_score(test_y, pred))
        # test_recall_list.append(recall_score(test_y, pred))
        # test_f1_list.append(f1_score(test_y, pred))


    result = pd.DataFrame({'roc': test_roc_list})
    # result = pd.DataFrame({'roc': test_roc_list,
    #                     'accuracy': test_acc_list,
    #                     'precision': test_prec_list,
    #                     'recall': test_recall_list,
    #                     'f1': test_f1_list})
    result.to_csv('gcn_aa.csv', header = True, index= False, encoding='utf-8-sig')
    