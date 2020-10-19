# -----------------------------------------------------------------------------
# Problem File
# Date:   09/04/2019
# Author: Craig Maxwell
# -----------------------------------------------------------------------------
# SYSTEM_TEST_01b
# -----------------------------------------------------------------------------

# --------------
# Imports
# --------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "System Test 01b"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":True, 
  "collisionAvoidance":True, 
  "plumeAvoidanceVehicle":True,
  "plumeAvoidanceObstacle":True,
  "finalConfigurationSelection":False}
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

# ---------------
# Model Variables
# ---------------
N = 2    # Number of dimensions
T = 21   # Number of time steps
T_end = 10 # End time (s)
V = 3    # Number of vehicles
P = 10   # Plume Length (m)
W = 1    # Plume Width (m)
M = 1000 # Big M Method
m = np.array([1, 1, 1]) # Mass of satellites (kg)
minApproachDist = 1 # Min displacement between satellie and any obstacle in each dimension (m)

# Initial state vector
x_ini = np.array([[-2, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])
# Final state vector
x_fin = np.array([[8, 0, 0, 0], [6, 0, 0, 0], [4, 0, 0, 0]])
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 10*np.ones([V, N]) # thrust (N)
# Objects - only work in 2D+
objects = np.array([[]])
# Safety distance
r = 1*np.ones([N])