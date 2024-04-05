import networkx as nx
import matplotlib.pyplot as plt


# plot polygon
def plot_polygon(ax, poly, label=None, **kwargs):
    x, y = poly.exterior.xy
    ax.fill(x, y, label=label, **kwargs)
    return


# custom figure set up
def set_figure(nc, nr,
               fs=10,
               fs_title=7.5,
               fs_legend=10,
               fs_xtick=3,
               fs_ytick=3,
               fs_axes=4,
               ratio=1):
    """
    Custom figure setup function that generates a nicely looking figure outline.
    It includes "making-sense"-fontsizes across all text locations (e.g. title, axes).
    You can always change things later yourself through the outputs or plt.rc(...).
    """

    fig, axs = plt.subplots(nc, nr, figsize=(fs*nr*ratio, fs*nc))

    try:
        axs = axs.flatten()
    except:
        pass

    plt.rc("figure", titlesize=fs*fs_title)
    plt.rc("legend", fontsize=fs*fs_legend)
    plt.rc("xtick", labelsize=fs*fs_xtick)
    plt.rc("ytick", labelsize=fs*fs_ytick)
    plt.rc("axes", labelsize=fs*fs_axes, titlesize=fs*fs_title)

    return fig, axs


def plot_graph(graph, ax,
               c_node='black', c_edge='black',  # coloring
               dw_edge=False, pos=None,  # edge type and node positioning
               node_size=10, edge_size=10):  # node and edge sizes

    """
    Plots the topological graph structure of a floor plan.
    Nodes can be colored based on the room category;
    Two possible edge types (if you want to show them in the first place):
    1) access connectivity (passage) e.g. by door; 2) adjacency e.g. by wall;
    Node positions are in 2D and could be for example the room centroids.
    """

    # Determine node position (if None is given)
    if not pos:
        pos = nx.spring_layout(graph, seed=7)  # random position for the nodes

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=c_node, ax=ax)

    # Draw edges
    if dw_edge:

        # Door connections
        edges = [(u, v) for (u, v, d) in graph.edges(data=True) if d["door"]]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=c_edge,
                               width=edge_size, ax=ax)

        # Adjacent connections
        edges = [(u, v) for (u, v, d) in graph.edges(data=True) if not d["door"]]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=c_edge,
                               width=edge_size, style="dashed", ax=ax)
    else:
        nx.draw_networkx_edges(graph, pos, edge_color=c_edge,
                               width=edge_size, ax=ax)