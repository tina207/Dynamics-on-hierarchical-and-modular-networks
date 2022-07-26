import numpy as np
import ast
import matplotlib.pyplot as plt
from numba import jit
from numba.typed import List

# time
dt = 0.01
sqdt = np.sqrt(dt)
t0 = 0
t_end = 60*5
t_steps = int((t_end - t0) / dt)

Delta = 0.5
sigma = 0.5
Delta_sq = Delta * Delta

#network
with open('modularGraph.txt') as f:
    neighbors_vector = f.read()

neighbors_vector = ast.literal_eval(neighbors_vector)
n_length = len(neighbors_vector)


@jit(nopython=True)
def do_simulation(network, omega, a, J, J_int, x=np.zeros((n_length, t_steps)), y=np.zeros((n_length, t_steps))):
    N = len(network)


    if np.all(x) == 0:
        # random initial values if there is none given
        x[:, 0] = np.random.rand(N) * 0.5
        y[:, 0] = np.random.rand(N) * 0.5

    # Radius of polar coordinates
    R = np.zeros(t_steps)

    for k in range(1, t_steps):
        z = 0

        for i in range(N):
            coup = 0

            for neigh in network[i]:
                coup += x[neigh, k - 1]

            # variables
            n_neighs = len(network[i])
            x_sq = x[i, k - 1] * x[i, k - 1]
            y_sq = y[i, k - 1] * y[i, k - 1]
            term = J_int * (1 - x_sq - y_sq)

            # noise
            r_gaussian = np.random.standard_normal(2)

            # neural mass models
            x[i, k] = ((0.5 * (
                    -2 * omega * y[i, k - 1] - x[i, k - 1] * (Delta_sq - term) - a * (1 - x_sq + y_sq))) + J * (
                               coup - x[i, k - 1] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[0] * sqdt + x[
                          i, k - 1]
            y[i, k] = ((0.5 * (2 * omega * x[i, k - 1] - y[i, k - 1] * (Delta_sq - term) + 2 * a * x[i, k - 1] * y[
                i, k - 1])) + J * (coup - y[i, k - 1] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[1] * sqdt + y[
                          i, k - 1]

            z += complex(x[i, k], y[i, k])

        z /= N
        R[k] = abs(z)

    return x, y, R


if __name__ == "__main__":

    with open('ERgraph.txt') as f:
        neighbors_vector = f.read()

    # numba can only work with lists
    neighbors_vector = ast.literal_eval(neighbors_vector)
    neighbors = List()
    for element in neighbors_vector:
        neighbors.append(element)


    x_new, y_new, r = do_simulation(network=neighbors, omega=1, a=0, J=0, J_int=1)

