import torch.nn

from dataset import Triplet_Dataset_RPLAN
from torch_geometric.loader import DataLoader
from model import GCN
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TripletMarginLoss

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

data_dir = 'external_data'

bs = 8
dataset = Triplet_Dataset_RPLAN(data_dir)

dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
data_batch = next(iter(dataloader))
print(len(dataset))
num_node_features = 5
hidden_dim = 64
embedding_dim = 1024

model = GCN(num_node_features, hidden_dim, embedding_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
triplet_loss = TripletMarginLoss(margin=1.0)

model.train()

num_epochs = 10
for epoch in range(num_epochs):
	epoch_loss = 0.0
	iteration = 0

	for anchor, positive, negative in dataloader:
		optimizer.zero_grad()

		anchor_embedding = model(anchor.to(device))
		positive_embedding = model(positive.to(device))
		negative_embedding = model(negative.to(device))

		loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		iteration += 1
		if iteration % 1000 == 0:
			print('Epoch [%d] [%d / %d] Average_Loss: %.5f' % (epoch + 1, iteration * bs, len(dataset), epoch_loss/(iteration * bs)))
	# Print the average loss for the epoch
	print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataset)}")
