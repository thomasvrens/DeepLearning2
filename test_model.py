import torch.nn

from dataset import Triplet_Dataset_RPLAN
from torch_geometric.loader import DataLoader
from model import GCN, CNN, GCN_CNN
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TripletMarginLoss

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

data_dir = 'external_data'

bs = 8
dataset = Triplet_Dataset_RPLAN(data_dir)

dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
data_batch = next(iter(dataloader))

num_node_features = 5
hidden_dim = 64
embedding_dim = 1024
CNN_output_dim = 9

model = CNN(CNN_output_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()

rand_embedding = torch.rand(1, 1024)
out = model(rand_embedding.to(device))
print(out)
print(len(out[0]))
print(out.size())