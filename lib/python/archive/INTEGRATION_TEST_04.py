# -----------------------------------------------------------------------------
# Problem File INTEGRATION_TEST_04
# INTEGRATION_TEST_04: Plume Impingement
# -----------------------------------------------------------------------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "Integration Test 4: Plume Impingement Constraints"
# Define active constraints
activeConstraints = {
  "basic":True, 
  "obstacleAvoidance":False, 
  "collisionAvoidance":True, 
  "plumeImpingement":True,
  "plumeAvoidanceVehicle":False,
  "plumeAvoidanceObstacle":False,
  "finalConfigurationSelection":False}
# Define outputs
outputs = {
  "dataframe":True,
  "paths":True,
  "inputVector":True,
  "separation":False,
  "minSeparation":False,
  "saveFig":False}
# Define colour scheme
colourScheme = "paletteDark6"

# ---------------
# Model Variables
# ---------------
N = 2        # Number of dimensions
T = 51       # Number of time steps
T_end = 100  # End time (s)
V = 2        # Number of vehicles
P = 10       # Plume Length (m)
W = 0.5        # Plume Width (m)
M = 1000     # Big M Method
TSFC = 1     # Thrust Specific Fuel Consumption (kg/N-s)
m = np.array([1, 1]) # Mass of satellites (kg)
minApproachDist = 1  # Min displacement between satellie and any obstacle in each dimension (m)
# Initial state vector
x_ini = np.array([[0, 0, 0, 0], [1, 0, 0, 0]])
# Final state vector
x_fin = np.array([[-10, 0, 0, 0], [1, 0, 0, 0]])
# State vector limits
x_lim = np.array([[1000, 1000, 100, 100], [1000, 1000, 100, 100]])
# Input vector limits (N)
u_lim = np.array([[10, 10], [10, 10]])
# Objects (Only N >= 2)
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance from objects in each axis
r = np.array([1, 1])
