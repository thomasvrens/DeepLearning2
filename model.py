import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

#embedding_dim = 1024
#embedding_dim = 16
#hidden_channels = 32
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, embedding_dim):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
        #return torch.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self, num_channels, hidden_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, hidden_channels, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(hidden_channels, 64, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GCN_CNN(nn.Module):
    def __init__(self, num_node_features, num_channels, num_classes):
        super(GCN_CNN, self).__init__()
        self.gcn = GCN(num_node_features, num_classes)
        self.cnn = CNN(num_channels, num_classes)
        self.fc = nn.Linear(num_classes * 2, num_classes)

    def forward(self, graph_data, image_data):
        gcn_out = self.gcn(graph_data)
        cnn_out = self.cnn(image_data)
        combined = torch.cat((gcn_out, cnn_out), dim=1)
        out = self.fc(combined)
        return out
