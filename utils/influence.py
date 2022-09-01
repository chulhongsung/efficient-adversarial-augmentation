import torch
from torch.autograd.functional import hessian #, hvp
from torch.autograd import grad

def compute_grad(model, sample, target, criterion):
    sample = sample.unsqueeze(0)
    prediction = model(sample)
    y = target.view(prediction.shape).to(torch.float64)
    loss = criterion(prediction.double(), (y+1)/2)
    return torch.autograd.grad(loss, list(model.parameters()))

def compute_sample_grads(model, data, targets, criterion):
    sample_grads = [compute_grad(model, data[i], targets[i], criterion) for i in range(data.shape[0])]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def compute_classifier_hessian(model, sample, target, criterion):

    def func(w, b):
        pred = sample @ w + b 
        loss = torch.sum(criterion(pred, (target+1)/2))
        return loss

    all_param = []
    
    for param in model.parameters():
        all_param.append(param.view(-1))
    
    hessian_list = hessian(func, (all_param[0], all_param[1]))

    tmp_hessian1 = torch.cat((hessian_list[0][0], hessian_list[0][1]), 1)
    tmp_hessian2 = torch.cat((hessian_list[1][0], hessian_list[1][1]), 1)
    hessian_mat = torch.cat((tmp_hessian1, tmp_hessian2), 0)

    return hessian_mat

def compute_loo_if(model, embedding, target, criterion):
    per_sample_grads = compute_sample_grads(model, embedding, target, criterion)

    grad_batch = torch.cat((per_sample_grads[0].squeeze(), per_sample_grads[1]), 1)
    hessian = compute_classifier_hessian(model, embedding, target, criterion)
    
    inv_hessian = torch.inverse(hessian)
    
    influence_list = []
    for i in range(grad_batch.shape[0]):
        tmp_influence = (grad_batch[i].unsqueeze(0) @ inv_hessian) @ grad_batch[i].unsqueeze(0).T
        influence_list.append(tmp_influence) 
    influence_arr = torch.abs(torch.cat(influence_list, axis=0).squeeze())
    
    return influence_arr

### Hessian Vector Product method

# total_x, total_y = next(iter(train_sample_loader))
# criterion = nn.BCEWithLogitsLoss(reduction = "none")

# def loss_hvp(param):
#     w = param[:300]
#     b = param[300:301]
#     pred = total_x @ w + b 
#     loss = torch.sum(criterion(pred.squeeze(), (total_y+1)/2))
#     return loss

# tmp_hvp = hvp(loss_hvp, torch.cat((all_param[0], all_param[1]), 0).unsqueeze(-1), sample_grad)[1]

### Test 
# assert influence_arr[0] == sample_grad.T @ tmp_hvp