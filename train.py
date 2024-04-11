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
from datetime import datetime
import numpy as np
 
cuda_available = torch.cuda.is_available()
torch.backends.mps.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

data_dir = 'external_data'

bs = 1
dataset = Triplet_Dataset_RPLAN(data_dir)

dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

num_node_features = 5
hidden_dim = 128
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


def moving_average(data, window_size):
    """
    Smooths the data using a moving average filter.

    Parameters:
    - data: The input data to be smoothed (a list or numpy array).
    - window_size: The number of data points to include in the moving average window.

    Returns:
    - smoothed_data: The smoothed data.
    """
    if window_size <= 1:
        return data  # No smoothing needed for window_size 1 or less

    # Ensure data is a numpy array for easy slicing
    data = np.array(data)

    # Pre-allocate smoothed_data array
    smoothed_data = np.zeros(len(data))

    # Calculate the moving average
    for i in range(len(data)):
        # Determine the window range
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        smoothed_data[i] = np.mean(data[start:end])

    return smoothed_data

num_epochs = 4
triplet_scaler = 0.01

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
			#print(anchor_embedding.size(), anchor_true_msk.size(), anchor_prediction.size())
			# if epoch > 2:
			triplet_loss = triplet_loss_fnct(anchor_embedding, positive_embedding, negative_embedding)
			trip_losses.append(triplet_loss.item())
			recon_loss = F.mse_loss(anchor_prediction, anchor_true_msk.to(device).float()) + F.mse_loss(positive_prediction, positive_true_msk.to(device).float()) + F.mse_loss(negative_prediction, negative_true_msk.to(device).float())
			#print(anchor_true_msk.size())

			# recon_loss = F.mse_loss(anchor_prediction, anchor_true_msk.to(device).float())
			#print(f'recon loss{recon_loss}')
			#print(f'triplet loss{triplet_loss}')
			loss = triplet_scaler * triplet_loss + recon_loss
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			iteration += 1

			losses.append(loss.item())
			# trip_losses.append(triplet_loss.item())
			recon_losses.append(recon_loss.item())

			if iteration % 1000 == 0:
				print('Epoch [%d] [%d / %d] Average_Loss: %.5f' % (epoch + 1, iteration * bs, len(dataset), epoch_loss/(iteration * bs)),end='  ')
				print(f'triplet loss {triplet_loss} -> {triplet_scaler * triplet_loss}, and recon loss {recon_loss}')
		# if epoch > 2:
		# 	triplet_scaler = 0.001
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

	plt.plot(losses, label = 'total loss', color = 'r')
	plt.plot(trip_losses, label = 'triplet loss', color = 'b', linestyle='--')
	plt.plot(recon_losses, label = 'reconstruction loss', color = 'g', linestyle='--')
	plt.title('Dual training loss over iterations')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 50
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 50)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 100
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 100)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 200
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 200)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 500
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 500)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 1000
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 1000)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 5000
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 5000)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 10000
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 10000)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 20000
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 20000)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 50000
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 50000)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

	window_size = 74743
	smoothed_loss = moving_average(losses, window_size)
	plt.plot(smoothed_loss)
	plt.title('Dual training loss over iterations (window = 74743)')
	plt.xlabel('Training Iteration')
	plt.ylabel('Loss')
	plt.show()

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

plt.plot(losses, label = 'total loss', color = 'r')
plt.plot(trip_losses, label = 'triplet loss', color = 'b', linestyle='--')
plt.plot(recon_losses, label = 'reconstruction loss', color = 'g', linestyle='--')
plt.title('Dual training loss over iterations')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 50
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 50)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 100
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 100)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 200
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 200)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 500
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 500)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 1000
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 1000)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 5000
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 5000)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 10000
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 10000)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 20000
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 20000)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 50000
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 50000)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()

window_size = 74743
smoothed_loss = moving_average(losses, window_size)
plt.plot(smoothed_loss)
plt.title('Dual training loss over iterations (window = 74743)')
plt.xlabel('Training Iteration')
plt.ylabel('Loss')
plt.show()
