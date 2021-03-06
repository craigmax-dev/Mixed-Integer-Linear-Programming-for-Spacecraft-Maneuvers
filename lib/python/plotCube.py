from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("equal")

# draw cube
r = [-1, 1]
r1 = [2, 8]
r2 = [2, 8]
r3 = [4, 6]

for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
    if (np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]) :
        ax.plot3D(*zip(s, e), color="b")

plt.show()