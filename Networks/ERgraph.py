import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

# Erdos-Renyi Network
# N = total number of nodes
# K = mean degree of each node
def makeERGraph(N, K):
    # probability
    P = float(K / N)
    # empty graph
    g = nx.empty_graph(N)

    # add edges
    for i in g.nodes():
        for j in g.nodes():
            if i < j:
                # random number
                R = random.random()
                # only if R< probability add edge from node i to j
                if R < P:
                    g.add_edge(i, j)

    adjMatrix = nx.adjacency_matrix(g)
    edges = nx.to_dict_of_lists(g)
    adjList = list(edges.values())

    return adjList, adjMatrix


if __name__ == '__main__':

    ERGraph, ER_adjMatrix = makeERGraph(1000, 15)
    np.set_printoptions(threshold=sys.maxsize)

    # format for Gephi
    solutions = np.argwhere(ER_adjMatrix == 1)
    solutions = solutions + 1
    with open('Gephi_ERGraph.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(solutions)

    # format for simulation
    stdoutOrigin = sys.stdout
    sys.stdout = open("../ERgraph.txt", "w")
    print(ERGraph)
    sys.stdout.close()
    sys.stdout = stdoutOrigin
    

    # check probability distribution, if generated correctly: Poisson distribution 
    # counting how many neighbors the node i has for histogram
    count = []
    for i in range(0, len(ERGraph)):
        count.append(len(ERGraph[i]))

    w = 1
    n, bins, patches = plt.hist(count, bins=np.arange(min(count), max(count) + w, w), density = True, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

    # colormap
    for i in range(len(patches)):
        patches[i].set_facecolor(plt.cm.PuBuGn(n[i] / max(n)))

    plt.title('Degree Distribution', fontsize=14)
    plt.xlabel('$k_i$', fontsize=13)
    plt.ylabel('$P(k_i)$', rotation=90, fontsize=13)
    plt.show()
