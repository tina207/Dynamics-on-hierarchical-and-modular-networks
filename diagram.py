from simulation import do_simulation, t_steps
import matplotlib.pyplot as plt
import numpy as np
import ast
from numba.typed import List

J_list = np.linspace(0, 2, 21)
with open('ERgraph.txt') as f:
    neighbors_vector = f.read()

# numba can only work with lists
neighbors_vector = ast.literal_eval(neighbors_vector)
neighbors = List()
for element in neighbors_vector:
    neighbors.append(element)

r = []
r_avg = []
r_var = []

# networks
for J in J_list:

    X, Y, R, _ = do_simulation(network=neighbors, omega=1, a=0, J=J, J_int=1)
    r.append(R)
    np.save("ERX_a=0", X)
    np.save("ERY_a=0", Y)
    np.save("ERr_a=0", r)
    r_avg.append(R.mean())
    r_var.append(R.var())


plt.figure()
plt.title("Erdos-Renyi", fontsize=60)
plt.xlabel("$J$", fontsize=58)
plt.ylabel("$<R>$", rotation=90, fontsize=58)
plt.plot(J_list, r_avg, color="cornflowerblue", marker="o", linewidth=3.0)
plt.xlim([0, 2])
plt.ylim([0, 1])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.show()

plt.figure()
plt.xlabel("$J$", fontsize=58)
plt.ylabel("$var(R)$", rotation=90, fontsize=58)
plt.plot(J_list, r_var, color="lightcoral", marker="o", label='Erdos-Renyi', linewidth=3.0)
plt.legend(fontsize=32, loc='upper left')
plt.xlim([0, 10])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.show()


#single oscillator
for J in J_list:
    X, Y, _, _ = do_simulation(network=neighbors, omega=1, a=0, J=0, J_int=J)
    if J == J_list[20]:
        x = X[-1]
        y = Y[-1]
    R = np.sqrt(X[-1]**2 + Y[-1]**2)
    np.save("X_singleO_a=0", X)
    np.save("Y_singleO_a=0", Y)
    np.save("R_singleO_a=0", R)
    r_avg.append(R.mean())


# Single Oscillator Plot
# R
plt.figure()
plt.title("$a = 1.2$", fontsize=60)
plt.xlabel("$J_{int}$", fontsize=58)
plt.ylabel("$<r>$", rotation=90, fontsize=58)
plt.plot(J_list, r_avg, color="lightgreen", marker="o", linewidth=2.0)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xlim([0, 2])
plt.ylim([0, 1])
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::2]))
for label in temp:
    label.set_visible(False)
plt.show()

# x, y
plt.figure()
plt.title("$a = 0$", fontsize=60)
plt.xlabel("$time$", fontsize=58)
plt.ylabel("$x(t)$", rotation=90, fontsize=58)
plt.plot(np.array(np.arange(stop=t_steps)), x, color="lightgreen", linewidth=2.0)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.xlim([0, 6000])
plt.ylim([-1, 1])
ax = plt.gca()
temp = ax.xaxis.get_ticklabels()
temp = list(set(temp) - set(temp[::2]))
for label in temp:
    label.set_visible(False)
plt.show()

