import torch.nn

from dataset import Triplet_Dataset_RPLAN
from torch_geometric.loader import DataLoader
from model import GCN, CNN
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import TripletMarginLoss
 
cuda_available = torch.cuda.is_available()
mps_available = print(torch.backends.mps.is_available())
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

GCN_model = GCN(num_node_features, hidden_dim, embedding_dim)
GCN_model = GCN_model.to(device)

CNN_model = CNN(9) # can add more inputs later, hard coded for now
CNN_model = CNN_model.to(device)

optimizer = optim.Adam(GCN_model.parameters(), lr=0.001)
triplet_loss = TripletMarginLoss(margin=1.0)

GCN_model.train()
CNN_model.train()

num_epochs = 10
for epoch in range(num_epochs):
	epoch_loss = 0.0
	iteration = 0

	for anchor, positive, negative in dataloader:
		optimizer.zero_grad()

		anchor_embedding = GCN_model(anchor.to(device))
		positive_embedding = GCN_model(positive.to(device))
		negative_embedding = GCN_model(negative.to(device))

		anchor_reconstruction = CNN_model(anchor_embedding)
		#positive_reconstruction = CNN_model(positive_embedding)
		#negative_reconstruction = CNN_model(negative_embedding)

		anchor_img, positive_img, negative_img = None, None, None

		triplet_loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
		#recon_loss = F.mse_loss(anchor_reconstruction, anchor_img) + F.mse_loss(positive_reconstruction, positiv_img) + F.mse_loss(negative_reconstruction, negative_img)
		recon_loss = F.mse_loss(anchor_reconstruction, anchor_img)


		loss = triplet_loss + recon_loss
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item()
		iteration += 1
		if iteration % 1000 == 0:
			print('Epoch [%d] [%d / %d] Average_Loss: %.5f' % (epoch + 1, iteration * bs, len(dataset), epoch_loss/(iteration * bs)))
	# Print the average loss for the epoch
	print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataset)}")
