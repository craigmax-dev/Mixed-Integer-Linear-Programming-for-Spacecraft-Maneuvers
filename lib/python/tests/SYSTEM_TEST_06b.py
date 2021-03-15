# -----------------------------------------------------------------------------
# INTEGRATION_TEST_06
# -----------------------------------------------------------------------------

# --------------
# Imports
# --------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "Integration Test 6: Plume Avoidance for Vehicles"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":True, 
  "collisionAvoidance":True, 
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
# Axis limits
axLims = np.array([0, 20, 0, 20, 0, 20])

# ---------------
# Model Variables
# ---------------
N = 3       # Number of dimensions
T = 20     # Number of time steps
T_end = 100 # End time (s)
V = 2       # Number of vehicles
P = 10       # Plume Length (m)
W = 2       # Plume Width (m)
M = 1000    # Big M Method
m = np.array([1, 1])  # Mass of satellites (kg)
minApproachDist = 1   # Min displacement between satellie and any obstacle in each dimension (m)
omega = 0 # (s) - only used in hills dynamics model

# Initial state vector
x_ini = np.array([[10, 10, 0, 0, 0, 0], [5, 5, 10, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[10, 10, 20, 0, 0, 0], [15, 15, 10, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 1*np.ones([V, N]) # Thrust (N)
# Objects - only work in 2D+
#objects = np.array([[2, 8, -2, 2, -6, 6], [2, 8, -6, 6, -2, 2]]) # [N1min, N1max, N2min, N2max, ...]
objects = np.array([[2, 18, 8, 12, 9, 11], [8, 12, 2, 18, 9, 11]]) #
# Safety distance
r = 1*np.ones([N])