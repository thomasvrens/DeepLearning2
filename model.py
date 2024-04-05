import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

#embedding_dim = 1024
#embedding_dim = 16
#hidden_channels = 32
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, embedding_dim)
    def forward(self, data):
        x, edge_index = data.geometry, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=0)
        x = self.linear(x)

        return x
        #return torch.log_softmax(x, dim=1)


