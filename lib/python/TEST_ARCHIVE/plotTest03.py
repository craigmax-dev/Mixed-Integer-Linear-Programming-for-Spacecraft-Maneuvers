import numpy as np
import math  as mt
V = 5

test = False

numCombinations = int(mt.factorial(V) / (2 * mt.factorial(V - 2)))
print(numCombinations)
combinations = np.empty([2, numCombinations])
for p in range(V):
	for q in range(V):
		if q > p:
			print("{} : {}".format(p, q))
					
print(combinations)