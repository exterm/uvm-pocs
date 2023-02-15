import random

import networkx as nx
import matplotlib.pyplot as plt

# replicating the results of the paper
# "Structure, Scaling, and Phase Transition in the Optimal Transport Network"
# by Steffen Bohn and Marcelo O. Magnasco, published 21 Feb 2007 in Physical Review Letters.

n = 3
gamma = 1
i0 = 1000

# triangular lattice graph in the shape of a hexagon, side length n (nodes)
def hexagon(n):
    # create row by row, starting from the top
    # note: rows have different lengths

    G = nx.Graph()

    # rows of increasing length
    for i in range(n):
        for j in range(n + i):
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

def draw_graph(G):
    # draw graph, with edge width proportional to conductance
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True,
        width=[G.edges[edge]['conductance']/total_conductance*100 for edge in G.edges()],
    )
    plt.show()

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
    # conductance of the edge is random
    G.edges[edge]['conductance'] = random.random()

# total conductance of the graph
total_conductance = sum([G.edges[edge]['conductance']**gamma for edge in G.edges()])**(1/gamma)

draw_graph(G)
