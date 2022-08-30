import torch
from torch.autograd.functional import hessian
from torch.autograd import grad

def compute_grad(model, sample, target, criterion):
    model.eval()    
    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    prediction = model(sample)
    y = target.view(prediction.shape).to(torch.float64)
    loss = criterion(prediction.double(), (y+1)/2)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(data.shape[0])]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads

def func(sample, target, w, b, criterion):
    pred = sample @ w + b 
    loss = torch.sum(criterion(pred, (target+1)/2))
    return loss

def compute_classifier_hessian(model)
    all_param = []
    
    for param in model.parameters():
        all_param.append(param.view(-1))
    
    hessian_list = torch.autograd.functional.hessian(func, (all_param[0], all_param[1]))

    tmp_hessian1 = torch.cat((hessian_list[0][0], hessian_list[0][1]), 1)
    tmp_hessian2 = torch.cat((hessian_list[1][0], hessian_list[1][1]), 1)
    hessian = torch.cat((tmp_hessian1, tmp_hessian2), 0)

    return hessian

def compute_loo_if(model, train_sample_loader):
    total_x, total_y = next(iter(train_sample_loader))
    per_sample_grads = compute_sample_grads(total_x, total_y)

    grad_batch = torch.cat((per_sample_grads[0].squeeze(), per_sample_grads[1]), 1)
    hessian = compute_classifier_hessian(model)
    
    influence_list = []
    for i in range(grad_batch.shape[0]):
        tmp_influence = (grad_batch[i].unsqueeze(0) @ hessian) @ grad_batch[i].unsqueeze(0).T
        influence_list.append(tmp_influence) 
    
    influence_arr = torch.abs(torch.cat(influence_list, axis=0).squeeze())
    return influence_arr

### Top K=50
# influence_arr = compute_loo_if(model,train_sample_loader)
# value, idx = torch.topk(influence_arr, k=50, axis=-1)

# np.unique(trn_batch_y.cpu().numpy(), return_counts=True)
# np.unique(trn_batch_y[idx].cpu().numpy(), return_counts=True)
# np.unique(total_y.cpu().numpy(), return_counts=True)