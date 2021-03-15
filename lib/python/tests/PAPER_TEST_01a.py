import numpy as np

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
title = "Paper Test 1: Multiple Vehicle Collision Avoidance"
# ---------------
# Model Variables
# ---------------
N = 2        # Number of dimensions
T = 21       # Number of time steps
T_end = 100  # End time (s)
V = 3        # Number of vehicles
P = 10       # Plume Length (m)
W = 5        # Plume Width (m)
M = 1000     # Big M Method
TSFC = 1     # Thrust Specific Fuel Consumption (kg/N-s)
m = np.array([1, 1, 1]) # Mass of satellites (kg)
minApproachDist = 1  # Min displacement between satellie and any obstacle in each dimension (m)
omega = 0 # (s) - only used in hills dynamics model

# --------------------
# Constraint Variables
# --------------------
# Problem File title
title = "Paper Test 01"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":False, 
  "collisionAvoidance":True, 
  "plumeImpingement":False,
  "plumeAvoidanceVehicle":False,
  "plumeAvoidanceObstacle":False,
  "finalConfigurationSelection":False}
# Define dynamics model
dynamicsModel = "freeSpace" # Define dynamics model - "hills" or "freeSpace"
# Define outputs
outputs = {
  "dataframe":True,
  "paths":True,
  "inputVector":True,
  "separation":False,
  "minSeparation":False,
  "saveFig":True}

# Define colour scheme
colourScheme = "paletteDark6"
axLims = np.array([-2, 8, -5, 5])


# Initial state vector
x_ini = np.array([[-2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])
# Final state vector
x_fin = np.array([[8, 0, 0, 0], [6, 0, 0, 0], [4, 0, 0, 0]])
# State vector limits
x_lim = np.array([[1000, 1000, 100, 100], [1000, 1000, 100, 100], [1000, 1000, 100, 100]])
# Input vector limits (N)
u_lim = np.array([[10, 10], [10, 10], [10, 10]])
# Objects (Only N >= 2)
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance from objects in each axis
r = np.array([1, 1, 1])
