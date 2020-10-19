import numpy as np

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
title = "Integration Test 4: Plume Impingement Constraints"
# ---------------
# Model Variables
# ---------------
N = 2        	# Number of dimensions
T = 51       	# Number of time steps
T_end = 100  	# End time (s)
V = 2        	# Number of vehicles
P = 10       	# Plume Length (m)
W = 5        	# Plume Width (m)
M = 1000     	# Big M Method
m = np.ones(V) 	# Mass of satellites (kg)
minApproachDist = 1  # Min displacement between satellie and any obstacle in each dimension (m)
#m = np.array([1, 1]) # Mass of satellites (kg)

# --------------------
# Constraint Variables
# --------------------
# Define active constraints
activeConstraints = { "basic":True, 
                      "obstacleAvoidance":False, 
                      "collisionAvoidance":True, 
                      "plumeImpingement":True,
                      "plumeAvoidanceVehicle":False,
                      "plumeAvoidanceObstacle":False,
                      "finalConfigurationSelection":False}
# Initial state vector
x_ini = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
# Final state vector
x_fin = np.array([[10, 0, 0, 0], [1, 0, 0, 0]])
# State vector limits
x_lim = 100 * np.ones((2*N, V))
# Input vector limits (N)
u_lim = np.array([[10, 10], [10, 10]])
# Objects (Only N >= 2)
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance from objects in each axis
r = np.array([1, 1])
