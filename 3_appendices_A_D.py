"""Code for appendix A: draw example graphs from SDA and SDAC models."""
# pylint: disable=redefined-outer-name,no-member
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
from networkx.algorithms import community
from sdnet import SDA
from sdnet.utils import make_dist_matrix, euclidean_dist
import _


HERE = os.path.dirname(os.path.realpath(__file__))
FIGS = os.path.join(HERE, 'figures')
if not os.path.exists(FIGS):
    os.makedirs(FIGS, exist_ok=True)

DPI = 150
FIGSIZE = (12, 12)
# PIXELS = 800


N = 500
NDIM = 2
CENTERS = 4
K = 15
ALPHA_SMALL = 2
ALPHA_LARGE = np.inf


# Routines --------------------------------------------------------------------

def make_graph(A):
    G = nx.from_numpy_array(A)
    communities = community.greedy_modularity_communities(G)
    for idx, comm in enumerate(communities, 1):
        for node_idx in comm:
            G.nodes[node_idx]['community'] = idx
            G.nodes[node_idx]['color'] = cm.Set1(idx-1)
    return G

def viz_space(ax, X, labels=None, edgecolors='#000000', linewidths=.6, **kwds):
    x, y = X[:, 0], X[:, 1]
    if labels is not None:
        labels = [ cm.Set1(x) for x in labels ]
    ax.scatter(x, y, c=labels, edgecolors=edgecolors, linewidths=linewidths,
               **kwds)
    ax.set_axisbelow(True)
    ax.grid(zorder=0, linestyle='--')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax

def viz_degseq(ax, degseq):
    ax.hist(degseq, color=cm.Set1(1), edgecolor='#000000', linewidth=1)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    return ax

def viz_sda_graph(ax, G, pos=None, with_labels=False, node_color=None,
                  node_border_color='#000000', node_border_width=.6,
                  edge_color='#000000', nodes_kws=None, edges_kws=None,
                  size_scaler=lambda x: np.sqrt(x)*2):
    if nodes_kws is None:
        nodes_kws = {}
    if edges_kws is None:
        edges_kws = {}
    if node_color is None:
        node_color = np.array([ n['color'] for n in G.nodes.values() ])
    node_size = np.array([ size_scaler(k) for k in G.degree() ])
    if pos is None:
        pos = nx.drawing.kamada_kawai_layout(G)
    nodes = \
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size,
                               with_labels=with_labels, ax=ax, **nodes_kws)
    nodes.set_edgecolor(node_border_color)
    nodes.set_linewidth(node_border_width)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, alpha=.2,
                           ax=ax, **edges_kws)
    ax.axis('off')
    return ax


# SDA model -------------------------------------------------------------------

# Set random seed
np.random.seed(44)

uniform, __ = _.simulate_uniform(N, NDIM)
lognormal, __ = _.simulate_lognormal(N, NDIM)
clusters, clusters_labs = _.simulate_normal_clusters(N, NDIM, CENTERS)

# Make plot
plt.close()
fig = plt.figure(figsize=FIGSIZE)
idx = 0
for X, labs in ((uniform, None), (clusters, clusters_labs), (lognormal, None)):
    D = make_dist_matrix(X, euclidean_dist, symmetric=True).astype(np.float32)
    for alpha in (None, ALPHA_SMALL, ALPHA_LARGE):
        idx += 1
        ax = plt.subplot(3, 3, idx)
        if alpha is None:
            ax = viz_space(ax, X, labs)
        else:
            sda = SDA.from_dist_matrix(D, alpha=alpha, k=K, directed=False)
            A = sda.adjacency_matrix(sparse=False)
            G = make_graph(A)
            ax = viz_sda_graph(ax, G)
        if idx <= 3:
            if alpha is None:
                title = '(a)'
            elif alpha == ALPHA_SMALL:
                title = '(b)'
            else:
                title = '(c)'
            ax.set_title(title)
plt.savefig(os.path.join(FIGS, 'fig-A-1.png'), dpi=DPI, bbox_inches='tight')


# SDC model -------------------------------------------------------------------

# Set random seed
np.random.seed(45)

# Social space
uniform, __ = _.simulate_uniform(N, NDIM)
# Degree sequences
poisson = _.simulate_degseq('poisson', N, K)
negbinom = _.simulate_degseq('negbinom', N, K)
powerlaw = _.simulate_degseq('powerlaw', N, K)


# Make plot
plt.close()
fig = plt.figure(figsize=FIGSIZE)
idx = 0
D = make_dist_matrix(uniform, euclidean_dist, symmetric=True).astype(np.float32)
for degseq in (poisson, negbinom, powerlaw):
    for alpha in (None, ALPHA_SMALL, ALPHA_LARGE):
        idx += 1
        ax = plt.subplot(3, 3, idx)
        if alpha is None:
            ax = viz_degseq(ax, degseq)
        else:
            sda = SDA.from_dist_matrix(D, alpha=alpha, k=K, directed=False)
            sda.set_degseq(degseq, sort=False)
            A = sda.conf_model(sparse=False)
            G = make_graph(A)
            ax = viz_sda_graph(ax, G)
        if idx <= 3:
            if alpha is None:
                title = '(a)'
            elif alpha == ALPHA_SMALL:
                title = '(b)'
            else:
                title = '(c)'
            ax.set_title(title)
plt.savefig(os.path.join(FIGS, 'fig-A-2.png'), dpi=DPI, bbox_inches='tight')
