import random

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

# replicating the results of the paper
# "Structure, Scaling, and Phase Transition in the Optimal Transport Network"
# by Steffen Bohn and Marcelo O. Magnasco, published 21 Feb 2007 in Physical Review Letters.

n = 8
gamma = 2
i0 = 1000
steps = 50

# triangular lattice graph in the shape of a hexagon, side length n (nodes)
def hexagon(n):
    # create row by row, starting from the top
    # note: rows have different lengths

    G = nx.Graph()

    # rows of increasing length
    for i in range(n):
        for j in range(n + i):
            G.add_node((i,j))
            if i > 0:
                if j < n + i - 1:
                    G.add_edge((i,j), (i-1,j))
                if j > 0:
                    G.add_edge((i,j), (i-1,j-1))
            if j > 0:
                G.add_edge((i,j), (i,j-1))

    # rows of decreasing length
    for i in range(n, 2*n-1):
        for j in range(3*n-2-i):
            G.add_edge((i,j), (i-1,j))
            G.add_edge((i,j), (i-1,j+1))
            if j > 0:
                G.add_edge((i,j), (i,j-1))


    # set positions as node attributes (node['pos])
    pos = {}
    for node in G.nodes():
        if node[0] < n:
            pos[node] = (node[0], node[1] - node[0]/2)
        else:
            pos[node] = (node[0], node[1] - (2*n - node[0])/2 + 1)

    nx.set_node_attributes(G, pos, 'pos')

    return G

def edge_width(G, edge):
    max_conductance = max([G.edges[edge]['conductance'] for edge in G.edges()])

    # normalize
    width = G.edges[edge]['conductance']/max_conductance

    # cut off very small values
    if width < 0.001:
        width = 0

    # sublinear scaling
    width = math.sqrt(width)

    # linear scaling
    width *= 10

    return width

def draw_graph(G, nodes=True):
    # draw graph, with edge width proportional to conductance
    pos = nx.get_node_attributes(G, 'pos')
    # make canvas a little taller than wide
    plt.figure(figsize=(6, 7))
    nx.draw(G, pos, with_labels=False,
        width=[edge_width(G, edge) for edge in G.edges()],
        # node size is absolute value of potential or 0 if nodes=False
        node_size=[abs(G.nodes[node]['potential']) if nodes else 0 for node in G.nodes()],
        # node color is sign of potential: green for positive, red for negative
        node_color=['g' if G.nodes[node]['potential'] > 0 else 'r' for node in G.nodes()],
    )
    plt.show()

def calculate_total_conductance(G):
    return sum([G.edges[edge]['conductance']**gamma for edge in G.edges()])**(1/gamma)

G = hexagon(n)

Gamma = 2*gamma/(gamma+1)
total_nodes = len(G.nodes())

# set current sources (negative sources are sinks)
for node in G.nodes():
    if node == (0,0):
        G.nodes[node]['source'] = i0
    else:
        G.nodes[node]['source'] = -i0/(total_nodes - 1)

# set conductances
for edge in G.edges():
    G.edges[edge]['conductance'] = random.random()

# total conductance of the graph
total_conductance = calculate_total_conductance(G)

# type sig:
def determine_potential(G: nx.Graph):
    '''
    determine the potential at each node based on the conductances and the current sources
    Use sources = (conductances_diagonal - conductances_offdiagonal) * potentials
    '''
    sources = np.array([G.nodes[node]['source'] for node in G.nodes()])
    conductances_offdiagonal = nx.adjacency_matrix(G, weight='conductance')

    conductances_diagonal = np.identity(len(G.nodes()))
    np.fill_diagonal(conductances_diagonal, conductances_offdiagonal.sum(axis=1))

    # solve the linear system
    potentials = np.linalg.solve(conductances_diagonal - conductances_offdiagonal, sources)

    return potentials

def step(G):
    # determine the potential at each node
    potentials = determine_potential(G)

    # set the potentials as node attributes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['potential'] = potentials[i]

    # current through each edge
    for edge in G.edges():
        G.edges[edge]['current'] = G.edges[edge]['conductance'] * (G.nodes[edge[0]]['potential'] - G.nodes[edge[1]]['potential'])

    # compute a new set of conductances by following the scaling
    for edge in G.edges():
        G.edges[edge]['conductance'] = abs(G.edges[edge]['current'])**(-(Gamma-2))

    # normalize conductances
    new_total_conductance = calculate_total_conductance(G)
    for edge in G.edges():
        G.edges[edge]['conductance'] *= total_conductance/new_total_conductance

for i in range(steps):
    step(G)

# draw_graph(G)
draw_graph(G, nodes=False)
