# -----------------------------------------------------------------------------
# Problem File
# Date:   25/01/2020
# Author: Craig Maxwell
# -----------------------------------------------------------------------------
# LIMITATION_TEST_02
# -----------------------------------------------------------------------------

# --------------
# Imports
# --------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "Limitation Test 02 - Collision Avoidance"
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
axLims = np.array([0, 10, -5, 5])

# ---------------
# Model Variables
# ---------------
N = 2       # Number of dimensions
T = 10      # Number of time steps
T_end = 10  # End time (s)
V = 2       # Number of vehicles
P = 10      # Plume Length (m)
W = 1       # Plume Width (m)
M = 1000    # Big M Method
m = np.array([1, 1])  # Mass of satellites (kg)
minApproachDist = 1   # Min displacement between satellie and any obstacle in each dimension (m)
dynamicsModel = "freeSpace" # Define dynamics model - "hills" or "freeSpace"
omega = 0 # (s) - only used in hills dynamics model

# Initial state vector
x_ini = np.array([[0, 0, 0, 0], [10, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[10, 0, 0, 0], [0, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 10*np.ones([V, N]) # Thrust (N)
# Objects - only work in 2D+
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance
r = 1*np.ones([N])