import torch.nn

from dataset import Triplet_Dataset_RPLAN
from torch_geometric.loader import DataLoader
from model import GCN, CNN, GCN_CNN
import torch.optim as optim
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch.nn import TripletMarginLoss
 
cuda_available = torch.cuda.is_available()
torch.backends.mps.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

data_dir = 'external_data'

bs = 1
dataset = Triplet_Dataset_RPLAN(data_dir)

dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

num_node_features = 5
hidden_dim = 64
embedding_dim = 1024
CNN_output_dim = 18

model = GCN_CNN(num_node_features, hidden_dim, embedding_dim, CNN_output_dim)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
triplet_loss_fnct = TripletMarginLoss(margin=1.0)

model.train()

losses = []
trip_losses = []
recon_losses = []

num_epochs = 10
try:
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		iteration = 0

		for anchor, positive, negative, anchor_true_msk, positive_true_msk, negative_true_msk in dataloader:
			optimizer.zero_grad()

			anchor_prediction = model(anchor.to(device))
			anchor_embedding = model.triplet_embedding

			positive_prediction = model(positive.to(device))
			positive_embedding = model.triplet_embedding

			negative_prediction = model(negative.to(device))
			negative_embedding = model.triplet_embedding

			triplet_loss = triplet_loss_fnct(anchor_embedding, positive_embedding, negative_embedding)
			#recon_loss = F.mse_loss(anchor_reconstruction, anchor_img) + F.mse_loss(positive_reconstruction, positiv_img) + F.mse_loss(negative_reconstruction, negative_img)
			#print(anchor_true_msk.size())

			recon_loss = F.mse_loss(anchor_prediction, anchor_true_msk.to(device).float())

			loss = triplet_loss + recon_loss
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			iteration += 1

			losses.append(loss.item())
			trip_losses.append(triplet_loss.item())
			recon_losses.append(recon_loss.item())

			if iteration % 1000 == 0:
				print('Epoch [%d] [%d / %d] Average_Loss: %.5f' % (epoch + 1, iteration * bs, len(dataset), epoch_loss/(iteration * bs)))
		# Print the average loss for the epoch
		print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataset)}")

except KeyboardInterrupt:

	# Pickle all the losses,
	pickle_dir = 'losses'
	total_save_str = f'{pickle_dir}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-total_losses.pickle'
	trip_save_str = f'{pickle_dir}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-triplet_losses.pickle'
	recon_save_str = f'{pickle_dir}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-recon_losses.pickle'

	with open(total_save_str, 'wb') as f:
		pickle.dump(losses, f)
	with open(trip_save_str, 'wb') as f:
		pickle.dump(trip_losses, f)
	with open(recon_save_str, 'wb') as f:
		pickle.dump(recon_losses, f)

	plt.plot(losses)
	plt.plot(trip_losses)
	plt.plot(recon_losses)
	plt.show()
