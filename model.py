import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F

#embedding_dim = 1024
#embedding_dim = 16
#hidden_channels = 32
class GCN(torch.nn.Module): #encoder
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

class CNN(nn.Module): #decoder
    def __init__(self, out_dims): #can add variable input dims
        super(CNN, self).__init__()

        # Need to add a spacial part to the 1024 dimension vector so that the CNN can work
        self.project_lin = nn.Linear(1024, 8*8*512)
        self.relu = nn.ReLU(True)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, out_dims, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.project_lin(x)
        x = self.relu(x)
        x = x.view(-1, 512, 8, 8)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = F.softmax(x, dim = 1) #double check that the dimension here is correct <--- DIM = 2 i think
        # Want to softmax over the channel dimension

        # Could still apply a softmax activation function here --> helpful with pixel classification
        # Dont think we need that though
        return x
