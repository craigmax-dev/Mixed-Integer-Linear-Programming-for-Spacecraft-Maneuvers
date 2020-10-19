import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a = np.delete(a, -1)
a = np.insert(a, 0, 11)
print(a)