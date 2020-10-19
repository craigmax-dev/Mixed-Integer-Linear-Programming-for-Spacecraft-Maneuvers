# Subtract two arrays and get maximum

import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([1, 3, 1, 2])

c = abs(a - b)
print(c)
print(max(c))