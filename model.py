import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class GCN(torch.nn.Module): #encoder
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(GCN, self).__init__()
        dropout_rate = 0.5
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(hidden_dim * 4, embedding_dim)
    def forward(self, data):
        x, edge_index = data.geometry.float(), data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        x = torch.mean(x, dim=0)
        x = self.linear(x)
        return x

class CNN(nn.Module): #decoder
    def __init__(self, in_features, out_dims): #can add variable input dims
        super(CNN, self).__init__()

        # Need to add a spacial part to the 1024 dimension vector so that the CNN can work
        # self.project_lin = nn.Linear(256, 8*8*512)
        self.project_lin = nn.Linear(in_features, 8 * 8 * in_features)
        self.relu = nn.ReLU(True)

        self.in_features = in_features

        # self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(in_features, in_features//2, kernel_size=4, stride=2, padding=1)
        #self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(in_features//2, in_features//4, kernel_size=4, stride=2, padding=1)
        #self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(in_features//4, in_features//8, kernel_size=4, stride=2, padding=1)
        # out dimes = 18
        #self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(in_features//8, in_features//16, kernel_size=4, stride=2, padding=1)
        self.deconv6 = nn.ConvTranspose2d(in_features//16, out_dims, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.project_lin(x)
        x = self.relu(x)
        # x = x.view(-1, 512, 8, 8)
        x = x.view(-1, self.in_features, 8, 8)

        # x = self.deconv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        x = self.deconv2(x)
        #x = self.bn2(x)
        x = self.relu(x)

        x = self.deconv3(x)
        #x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv4(x)
        #x = self.bn4(x)
        x = self.relu(x)

        #x = F.interpolate(x, scale_factor=2, mode='nearest')  # Resizing to double the dimensions

        x = self.deconv5(x)
        x = self.relu(x)

        x = self.deconv6(x)
        x = F.softmax(x, dim = 1) #double check that the dimension here is correct <--- DIM = 2 i think
        # Want to softmax over the channel dimension

        x = x.permute(0, 2, 3, 1)
        # Could still apply a softmax activation function here --> helpful with pixel classification
        return x

class GCN_CNN(nn.Module): #decoder
    def __init__(self, input_dim, hidden_dim, embedding_dim, CNN_output_dim): #can add variable input dims
        super(GCN_CNN, self).__init__()
        self.GCN = GCN(input_dim, hidden_dim, embedding_dim)
        self.CNN = CNN(embedding_dim, CNN_output_dim)
        self.triplet_embedding = None
    def forward(self, x):
        x = self.GCN(x)
        self.triplet_embedding = x
        x = self.CNN(x)
        return x
