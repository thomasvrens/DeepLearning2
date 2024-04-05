import networkx as nx
import torch
import os
import random
import torch_geometric as pyg
from utils import load_pickle

class Triplet_Dataset_RPLAN(torch.utils.data.Dataset):

    """
    Geometry Triplet Graph Dataset for RPLAN.

    Generates a graph pair as a single Data() object.
    edge_index:
        room_i to room_j: all combinations, except i == j, should be in there.
        (NOT THE CASE ANYMORE; FOR MSD ONLY ACCESS CONNECTED NODES ARE IN THERE!)
    node_features:
        'geometry': geometric features of rooms (based on bounding boxes)
        'order': whether graph 0 or 1, such that the graph matching network knows which to match
    edge_features:
        'inter-geometry': inter-geometric features between rooms (based on 'difference' in bounding boxes).
            Note: this is not symmetrical. Therefore, the geometry graph is a directed graph.
    """

    def __init__(self,
                 dir_rplan,
                 graph_path='gmn-graph',
                 triplet_path='triplets_iou_74K.pickle',
                 shuffle=True,
                 mode='train'):

        self.graph_path = os.path.join(dir_rplan, graph_path)
        self.triplets = load_pickle('train_triplets_iou_74K.pickle')
        if shuffle: random.shuffle(self.triplets)
        # self.image_transform = image_transform
        # self.graph_transform = graph_transform

    def __getitem__(self, index):
        # TODO: graph transformations (??)

        # get identity pairs (saved as tuples of two integers)
        triplet = self.triplets[index]

        # get graphs (saved as networkx DiGraph() objects)
        # also: add order index ("0" for graph 1; "1" for graph 2)
        graphs = []
        for i, id in enumerate(triplet):  # actually a 4-tuple: [a, p, a, n]
            graph = torch.load(os.path.join(self.graph_path, f'{id}'))
            graph.add_nodes_from([[n, {'order': i, 'id': id}] for n in graph.nodes()])
            graphs.append(graph)

        # make large disconnected graph through unification of the two graphs
        graph_triplet = nx.disjoint_union_all(graphs)

        # make pytorch geometric graph
        graph_triplet = pyg.utils.from_networkx(graph_triplet)

        return graph_triplet

    def __len__(self):
        return len(self.triplets)