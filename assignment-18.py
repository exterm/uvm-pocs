import networkx as nx
import matplotlib.pyplot as plt

n = 8

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

G = hexagon(n)

# draw graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True)
plt.show()
