from simulation import do_simulation, t_steps
import matplotlib.pyplot as plt
import numpy as np
import ast
from numba.typed import List

J_list = np.linspace(0, 2, 20)

with open('modularGraph.txt') as f:
    neighbors_vector = f.read()

# numba can only work with lists
neighbors_vector = ast.literal_eval(neighbors_vector)
neighbors = List()
for element in neighbors_vector:
    neighbors.append(element)

r = []
r_avg = []

# networks
for J in J_list:
    X, Y, R = do_simulation(network=neighbors, omega=1, a=0.95, J=J, J_int=1)
    r.append(R)
    np.save("X_a=0.95", X)
    np.save("Y_a=0.95", Y)
    np.save("r_a=0.95", r)
    r_avg.append(np.average(R[5000:]))


'''
#single oscillator
for J in J_list:
    X, Y, _ = do_simulation(network=neighbors, omega=1, a=1.2, J=0, J_int=J)
    R = np.sqrt(X[-1]**2 + Y[-1]**2)
    np.save("X_singleO_a=1.2", X)
    np.save("Y_singleO_a=1.2", Y)
    np.save("R_singleO_a=1.2", R)
    r_avg.append(np.average(R[5000:]))
'''

plt.title("sigma=0.5, a=0.95, Delta=0.5, omega=1")
plt.xlabel("J")
plt.ylabel("r_avg")
plt.plot(J_list, r_avg, color="red", marker="o")
plt.legend()
plt.show()

