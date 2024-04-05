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
        x, edge_index = data.geometry.float(), data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=0)
        x = self.linear(x)

        return x

class CNN(nn.Module):
    def __init__(self, embedding_dim, initial_shape, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.ConvTranspose2d(initial_shape[0], 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
