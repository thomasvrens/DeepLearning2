import pickle
import numpy as np
import torch_geometric
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import os
from collections import OrderedDict


# Loading and unloading pickled files
def save_pickle(object, filename):
    """
    Saves a pickled file.
    """
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    """
    Loads a pickled file.
    """
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


# Floor plan colorization
def colorize_floorplan(img, classes, cmap):

    """
    Colorizes an integer-valued image (multi-class segmentation mask)
    based on a pre-defined cmap color set.
    """

    h, w = np.shape(img)
    img_c = (np.ones((h, w, 3)) * 255).astype(int)
    for cat in classes:
        color = np.array(cmap(cat))[:3] * 255
        img_c[img == cat, :] = color.astype(int)

    return img_c


# Graph-based utilities
def pyg_to_nx(graph, node_attrs, edge_attrs, graph_attrs):
    """
    Creates a networkx graph from a pytorch geometric graph data structure
    :param edge_attrs: transferred edge attributes
    :param node_attrs: transferred node attributes
    :param graph: a pytorch geometric graph structure
    :return: graph with fewer node and edge categories
    """
    return torch_geometric.utils.to_networkx(graph, to_undirected=True,
                                             node_attrs=node_attrs,
                                             edge_attrs=edge_attrs,
                                             graph_attrs=graph_attrs)


def simple_graph(graph):
    """
    Creates simple graph with reduced node and edge attributes.
    Remaining node attributes: categorical room types. Should be called 'category'.
    Remaining edge attributes: door types. Should be called 'door'
    """

    # Creates simple Networkx graph with only 'category' and 'door' attributes.
    graph = torch_geometric.utils.to_networkx(graph,
                                              to_undirected=True,
                                              node_attrs=['category'],
                                              edge_attrs=['door'])

    return graph


def remove_attributes_from_graph(graph, list_attr=['polygons']):
    """
    Removes attributes from graph.
    :param graph: Input topological graph.
    :param list_attr: Attributes to-be removed.
    :return: Output topological graph with removed attributes.
    """

    for attr in list_attr:
        for n in graph.nodes(): # delete irrelevant nodes
            del graph.nodes[n][attr]
    return graph


# "Random" utilities
def weighted_average(a, b, c):
    """
    Computes a weighted average.
    """
    return (a * c + b) / (c + 1)


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adjust_learning_rate(optimizer, init_lr, epoch, nr_epochs):
    """Cosine decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / nr_epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, alpha=0.1):
        self.moving_avg = alpha * val + (1-alpha) * self.val
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    torch.save(state, f'{filename}.pth.tar')

def load_checkpoint(model, filename):
    filename = f'{filename}.pth.tar'
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename,
            map_location=lambda storage, loc: storage.cuda() if torch.cuda.is_available() else storage.cpu())

        state_dict = checkpoint['state_dict']

        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v

        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}'"
                .format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))


# helper functions for GMN
def reshape_and_split_tensors(graph_feats, n_splits):
    feats_dim = graph_feats.shape[-1]
    graph_feats = torch.reshape(graph_feats, [-1, feats_dim * n_splits])
    graph_feats_splits = []
    for i in range(n_splits):
        graph_feats_splits.append(graph_feats[:, feats_dim * i: feats_dim * (i + 1)])
    return graph_feats_splits