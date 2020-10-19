import numpy as np

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
title = "Integration Test 7: Obstacle Constraints"
# ---------------
# Model Variables
# ---------------
N = 2       # Number of dimensions
T = 101      # Number of time steps
T_end = 50 # End time (s)
V = 1       # Number of vehicles
P = 1       # Plume Length (m)
W = 1       # Plume Width (m)
M = 1000    # Big M Method
m = np.array([1, 1]) # Mass of satellites (kg)
TSFC = 1    # Thrust Specific Fuel Consumption (kg/N-s)
minApproachDist = 1 # Min displacement between satellie and any obstacle in each dimension (m)

# --------------------
# Constraint Variables
# --------------------
# Define active constraints
activeConstraints = {	"basic":True, 
											"obstacleAvoidance":True, 
											"collisionAvoidance":True, 
											"plumeImpingement":False,
											"plumeAvoidanceVehicle":False,
											"plumeAvoidanceObstacle":False,
											"finalConfigurationSelection":False}
# Initial state vector
x_ini = np.array([[0, 0, 0, 0]])
# Final state vector
x_fin = np.array([[20, 0, 0, 0]])
# State vector limits
x_lim = np.array([[1000, 1000, 100, 100], [1000, 1000, 100, 100]])
# Input vector limits
u_lim = np.array([[10, 10]]) # thrust (N)
# Objects - only work in 2D+
objects = np.array([[4, -10, 6, 2], [10, -2, 12, 10]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance
r = np.array([1, 1]) # In each axis