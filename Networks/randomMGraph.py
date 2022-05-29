import numpy as np
import numpy.random
import csv


# generate edges between modules randomly
# partition: list containing the indices of the nodes in each module
# E: number of edges network will have
def make_edge(adjmatrix, E, partition):
    if len(partition) < 2.0:
        raise TypeError("There should be at least two modules")

    # number of modules
    mods = len(partition)
    # number of nodes in a module
    Nmod = len(partition[0])

    counter = 0
    while counter < E:
        # two random modules
        mod1 = int(mods * numpy.random.rand())
        mod2 = int(mods * numpy.random.rand())

        # if nodes are in the same module, choose other
        if mod1 == mod2:
            continue

        # random nodes from chosen modules
        node1idx = int(Nmod * numpy.random.rand())
        node2idx = int(Nmod * numpy.random.rand())
        node1 = partition[mod1][node1idx]
        node2 = partition[mod2][node2idx]

        # if edge already exists, jump to beginning
        if adjmatrix[node1, node2]:
            continue

        # set adjacency matrix to 1 for new edge
        adjmatrix[node1, node2] = 1
        adjmatrix[node2, node1] = 1
        counter += 1


# Graph with randomly connected modules
# k-list = list of mean degree of each level
def randomMGraph(shape, klist, outdtype=np.uint8):
    if len(shape) != len(klist):
        raise ValueError("Shape and k-list not aligned.")
    if shape[0] < 1:
        raise ValueError("First hierarchical level (Shape[0]) must contain more than one module.")

    # total number of nodes
    N = np.multiply.reduce(shape)

    # number of hierarchical levels
    levels = len(shape)

    adjmatrix = np.zeros((N, N), outdtype)

    for level in range(levels):

        # nr of modules at a level = block:

        # only one block, jump to beginning
        if shape[level] == 1:
            continue

        # most outer module, whole graph
        if level == 0:
            nblocks = 1
        else:
            # nblocks = all modules until current level
            nblocks = int(np.multiply.reduce(shape[:level]))

        # edges within blocks
        for b in range(nblocks):
            # find the indices for each module:

            # number of nodes in one block
            Nblock = N // nblocks
            # min index of a node in current module
            iminblock = b * Nblock

            # number of nodes in a block in current level
            nmod = shape[level]
            Ns = Nblock // nmod

            partition = []

            # append all indices of nodes of a module
            for p in range(nmod):
                partition.append(range(iminblock + p * Ns, iminblock + (p + 1) * Ns))

            # number of edges (undirected)
            nE = 0.5 * Nblock * klist[level]

            make_edge(adjmatrix, nE, partition)

    return adjmatrix


if __name__ == "__main__":

    # shape: 4 modules, divided further into 4 modules, each 16 nodes, total nodes: 4*4*16=256
    randomMGraph = randomMGraph([4, 4, 16], [1, 4, 13])

    # format for gephi
    solutions = np.argwhere(randomMGraph == 1)
    solutions = solutions + 1

    with open('Gephi_randomGraph.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerows(solutions)
