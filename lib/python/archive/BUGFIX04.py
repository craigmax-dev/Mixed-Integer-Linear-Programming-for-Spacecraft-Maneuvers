import numpy as np

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
title = "Case 1: 1D Optimization"
# ---------------
# Model Variables
# ---------------
N = 2       # Number of dimensions
T = 100     # Number of time steps
T_end = 100 # End time (s)
V = 2       # Number of vehicles
P = 10      # Plume Length (m)
W = 1       # Plume Width (m)
M = 100     # Big M Method
m = np.array([1, 1]) # Mass of satellites (kg)
TSFC = 1    # Thrust Specific Fuel Consumption (kg/N-s)
minApproachDist = 0.5 # Min displacement between satellie and any obstacle in each dimension (m)

# --------------------
# Constraint Variables
# --------------------
# Initial state vector
x_ini = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[-10, 0, 0, 0], [10, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = np.array([[1000, 1000, 100, 100], [1000, 1000, 100, 100]]) # x_max = 2, v_max = 10
# Input vector limits
u_lim = np.array([[10, 10], [10, 10]]) # thrust (N)
# Objects - only work in 2D+
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance
r = np.array([1, 1]) # In each axis
