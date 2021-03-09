import numpy as np

# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
title = "System Test 1: Multiple Vehicle Collision Avoidance"
# ---------------
# Model Variables
# ---------------
N = 3       # Number of dimensions
T = 100     # Number of time steps
T_end = 100 # End time (s)
V = 1       # Number of vehicles
P = 6       # Plume Length (m)
W = 1       # Plume Width (m)
M = 1000    # Big M Method
m = 1*np.ones([N])  # Mass of satellites (kg)
minApproachDist = 1   # Min displacement between satellie and any obstacle in each dimension (m)
omega = 7200 # (s) - only used in hills dynamics model

# --------------------
# Constraint Variables
# --------------------
# Problem File title
title = "System Test 01"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":True, 
  "collisionAvoidance":True, 
  "plumeImpingement":True,
  "plumeAvoidanceVehicle":False,
  "plumeAvoidanceObstacle":True,
  "finalConfigurationSelection":False}
# Define dynamics model
# dynamicsModel = "freeSpace" # Define dynamics model - "hills" or "freeSpace"
dynamicsModel = "hills" # Define dynamics model - "hills" or "freeSpace"
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
axLims = np.array([-10, 10, -10, 10, -10, 10])

# Initial state vector
x_ini = np.array([[-3, -3, -3, 0, 0, 0]]) # x_i = 1, v_i = 0
# x_ini = np.array([[10, 10, 0, 0, 0, 0], [5, 5, 10, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[3, 3, 3, 0, 0, 0]]) # x_i = 1, v_i = 0
# x_fin = np.array([[10, 10, 20, 0, 0, 0], [15, 15, 10, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 0.01*np.ones([V, N]) # Thrust (N)
# Objects - only work in 2D+
objects = np.array([[-2, 2, -2, 2, -2, 2]]) #
# objects = np.array([[2, 18, 8, 12, 9, 11], [8, 12, 2, 18, 9, 11]]) #
# Safety distance
r = 1*np.ones([N])
