from dataset import Triplet_Dataset_RPLAN
from torch_geometric.loader import DataLoader

data_dir = 'external_data'

bs = 4
dataset = Triplet_Dataset_RPLAN(data_dir)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
data_batch = next(iter(dataloader))
print(data_batch)
