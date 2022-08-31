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
t_relax = t_end

t_steps = int((t_end - t0) / dt)
t_relax_steps = int((t_relax - t0) / dt)

Delta = 0.5
sigma = 0.5
Delta_sq = Delta * Delta

#network
with open('ERgraph.txt') as f:
    neighbors_vector = f.read()

neighbors_vector = ast.literal_eval(neighbors_vector)
n_length = len(neighbors_vector)


@jit(nopython=True)
def do_simulation(network, omega, a, J, J_int):
    N = len(network)

    # relaxation time for simulation
    x_relax = np.random.rand(N)
    y_relax = np.random.rand(N)

    # Radius of polar coordinates
    R = np.zeros(t_steps)
    # Global phase coherence (obtained from Kuramoto order parameter)
    r_global = np.zeros(t_steps)

    for k in range(1, t_relax_steps):
        x_old = x_relax
        y_old = y_relax

        for i in range(N):
            coup_x = 0
            coup_y = 0

            for neigh in network[i]:
                coup_x += x_old[neigh]
                coup_y += y_old[neigh]

            # variables
            n_neighs = len(network[i])
            x_sq = x_old[i] * x_old[i]
            y_sq = y_old[i] * y_old[i]
            term = J_int * (1 - x_sq - y_sq)

            # noise
            r_gaussian = np.random.standard_normal(2)

            # neural mass models for every n (we do not have to save every timestep)
            x_relax[i] = ((0.5 * (
                    -2 * omega * y_old[i] - x_old[i] * (Delta_sq - term) - a * (1 - x_sq + y_sq))) + J * (
                               coup_x - x_old[i] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[0] * sqdt + x_old[
                          i]
            y_relax[i] = ((0.5 * (2 * omega * x_old[i] - y_old[i] * (Delta_sq - term) + 2 * a * x_old[i] * y_old[
                i])) + J * (coup_y - y_old[i] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[1] * sqdt + y_old[
                          i]

    # Initialize
    x = np.zeros((N, t_steps))
    y = np.zeros((N, t_steps))

    # Relaxed values as initial values for real simulation
    x[:, 0] = x_relax
    y[:, 0] = y_relax

    # Initialize average Z
    z = x_relax + 1.0j * y_relax
    z = z.mean() 
    R[0] = abs(z)

    # Initialize average Kuramoto
    psi = np.arctan2(y_relax, x_relax)
    kuramoto = np.exp(1.0j * psi)
    kuramoto = kuramoto.mean()
    r_global[0] = abs(kuramoto)

    for k in range(1, t_steps):
        z = 0.0
        kuramoto = 0.0

        for i in range(N):
            coup_x = 0
            coup_y = 0

            for neigh in network[i]:
                coup_x += x[neigh, k - 1]
                coup_y += y[neigh, k - 1]

            # variables
            n_neighs = len(network[i])
            x_sq = x[i, k - 1] * x[i, k - 1]
            y_sq = y[i, k - 1] * y[i, k - 1]
            # J_int is internal scaling factor
            term = J_int * (1 - x_sq - y_sq)

            # noise
            r_gaussian = np.random.standard_normal(2)

            # neural mass models
            # J is global scaling factor
            x[i, k] = ((0.5 * (
                    -2 * omega * y[i, k - 1] - x[i, k - 1] * (Delta_sq - term) - a * (1 - x_sq + y_sq))) + J * (
                               coup_x - x[i, k - 1] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[0] * sqdt + x[
                          i, k - 1]
            y[i, k] = ((0.5 * (2 * omega * x[i, k - 1] - y[i, k - 1] * (Delta_sq - term) + 2 * a * x[i, k - 1] * y[
                i, k - 1])) + J * (coup_y - y[i, k - 1] * n_neighs) / n_neighs) * dt + sigma * r_gaussian[1] * sqdt + y[
                          i, k - 1]


            z += x[i, k] + 1.0j * y[i, k]
            psi = np.arctan2(y[i, k], x[i, k])
            kuramoto += np.exp(1.0j * psi)

        z /= N
        kuramoto /= N

        R[k] = abs(z)
        r_global[k] = abs(kuramoto)

    return x, y, R, r_global


if __name__ == "__main__":
    '''
    with open('CHM.txt') as f:
        neighbors_vector = f.read()

    # numba can only work with lists
    neighbors_vector = ast.literal_eval(neighbors_vector)
    neighbors = List()
    for element in neighbors_vector:
        neighbors.append(element)


    x_new, y_new, r_new, r_global_new = do_simulation(network=neighbors, omega=1, a=0.95, J=0, J_int=1)
 
    plt.figure()
    for j in range(5):
        plt.plot(np.arange(y_new[0,:].size), x_new[j, :]*x_new[j, :] +  y_new[j, :]*y_new[j, :] )
    plt.show()

    plt.figure()
    plt.plot(r)
    plt.show()

    print(r.mean())
    '''
