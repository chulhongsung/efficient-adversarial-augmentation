import torch

class Linear_predictor(torch.nn.Module):

    def __init__(self, emb_dim, num_tasks):
        super(Linear_predictor, self).__init__()
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)
        # self.bn = nn.BatchNorm1d(emb_dim)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        # return self.graph_pred_linear(self.bn(x))
        return self.graph_pred_linear(x)
        
        # return self.act(self.graph_pred_linear(self.bn(x)))