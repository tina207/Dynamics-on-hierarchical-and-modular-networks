import numpy as np
import numpy.random
import sys


# helper to generate modules with preferential attachment
def generateModule(E, partition, cprob):

    if len(partition) < 1:
        raise ValueError("Less than one module, generating a random graph")

    # number of modules
    mods = len(partition)
    # number of nodes in a module
    Nmod = len(partition[0])
    # total
    N = mods * Nmod

    blockmatrix = np.zeros((N, N), np.uint8)
    counter = 0
    while counter < E:
        # two random modules
        mod1 = int(mods * numpy.random.rand())
        mod2 = int(mods * numpy.random.rand())

        # if nodes are in the same module, choose other
        if mod1 == mod2:
            continue


        # connect two nodes given the cumulative probabilities
        x = numpy.random.rand()  # random number between 0 and 1
        xsum = np.sum(np.sign(cprob - x))
        idx = int(0.5 * (Nmod - xsum))
        node1 = mod1 * Nmod + idx

        x = numpy.random.rand()
        xsum = np.sum(np.sign(cprob - x))
        idx = int(0.5 * (Nmod - xsum))
        node2 = mod2 * Nmod + idx

        # if same node or edge already exists, jump to beginning without incrementing counter
        if node1 == node2:
            continue
        if blockmatrix[node1, node2]:
            continue

        # set matrix to 1 for new edge
        blockmatrix[node1, node2] = 1
        blockmatrix[node2, node1] = 1
        counter += 1

    return blockmatrix

# generate edges at random between nodes with a probability
def randomEdges(E, cprob):

    Nmod = len(cprob)

    modmatrix = np.zeros((Nmod, Nmod), np.uint8)

    counter = 0
    while counter < E:
        # connect two nodes given cumulative probabilities
        x = numpy.random.rand()
        xsum = np.sum(np.sign(cprob - x))
        node1 = int(0.5 * (Nmod - xsum))

        x = numpy.random.rand()
        xsum = np.sum(np.sign(cprob - x))
        node2 = int(0.5 * (Nmod - xsum))

        # if same node or edge already exists, jump to beginning
        if node1 == node2:
            continue
        if modmatrix[node1, node2]:
            continue

        # set matrix to 1 for new edge
        modmatrix[node1, node2] = 1
        modmatrix[node2, node1] = 1
        counter += 1

    return modmatrix


# model of hierarchical and modular networks after Zamora-Lopez

def modularGraph(shape, klist, gammalist, outdtype=np.uint8):

    if len(shape) != len(klist):
        raise ValueError("Shape and k-list not aligned.")

    if len(gammalist) != len(shape):
        raise ValueError( "Shape and gamma-list are not aligned." )

    # total number of nodes
    N = np.multiply.reduce(shape)
    nlevels = len(shape)
    adjmatrix = np.zeros((N, N), outdtype)

    # calculating alpha according to formula
    alpha = 1.0 / (np.array(gammalist, float) - 1.0)

    for level in range(nlevels-1):
        # find the number of blocks, modules and nodes

        # Number of blocks
        if shape[level] == 1:
            continue
        if level == 0:
            nblocks = 1
        else:
            nblocks = int(np.multiply.reduce(shape[:level]))

        # Number of nodes per block
        Nblock = np.multiply.reduce(shape[level:])

        # Number of modules per block
        mods = shape[level]

        # Number of nodes per module
        Nmod = Nblock // mods


        # append all indices of nodes of a module
        partition = np.zeros((mods, Nmod), np.uint)
        for i in range(mods):
            partition[i] = np.arange(i*Nmod, (i+1)*Nmod, dtype=np.uint)


        # if current level not one of the two last hierarchical levels
        if level < nlevels - 2:
            nodeweights = np.ones(Nmod, float)
            nmodsnext = shape[level+1]
            Nmodnext = Nmod // nmodsnext

            # assigning weight like scale-free networks
            for i in range(nmodsnext):
                nodeweights[i*Nmodnext:(i*Nmodnext+Nmodnext)] = ((np.arange(Nmodnext) +1 ).astype(float))**-alpha[level]
            # Probability of a node to be chosen
            nodeweights /= nodeweights.sum()
            cprob = nodeweights.cumsum()
        else:
            nodeweights = np.ones(Nmod, float)
            nodeweights = ((np.arange(Nmod) + 1).astype(float))**-alpha[level]

            nodeweights /= nodeweights.sum()
            cprob = nodeweights.cumsum()


        # number of edges per block
        Eblock = 0.5 * Nblock * klist[level]

        # edges between modules in the current hierarchical level
        for b in range(nblocks):
            minidx = b*Nblock
            maxidx = (b+1)*Nblock
            adjmatrix[minidx:maxidx, minidx:maxidx] = generateModule(Eblock, partition, cprob)

    # last hierarchical level
    Nmod = shape[-1]
    mods = len(adjmatrix) // Nmod

    # weights for nodes within modules
    nodeweights = ((np.arange(Nmod)+1).astype(float))**(-alpha[-1])
    nodeweights /= nodeweights.sum()
    cprob = nodeweights.cumsum()

    # number of edges per module
    Emod = 0.5 * Nmod * klist[-1]

    # edges between modules in current hierarchical level
    for i in range(mods):
        minidx = i * Nmod
        maxidx = (i+1) * Nmod

        adjmatrix[minidx:maxidx, minidx:maxidx] = randomEdges(Emod, cprob)

    return adjmatrix


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)

    exampleGraph = modularGraph([2,10,50], [5,10,30], [2,3,3], outdtype=np.uint8)
    solutions = np.argwhere(exampleGraph == 1)
    solutions = solutions + 1
    stdoutOrigin = sys.stdout
    sys.stdout = open("Gephi_modularGraph.txt", "w")
    print("source,target")
    print(str(solutions).replace(' [', '').replace('[', '').replace(']', ''))

    sys.stdout.close()
    sys.stdout = stdoutOrigin
