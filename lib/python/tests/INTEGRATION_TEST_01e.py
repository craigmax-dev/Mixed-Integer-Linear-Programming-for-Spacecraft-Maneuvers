# -----------------------------------------------------------------------------
# Problem File
# Date:   09/04/2019
# Author: Craig Maxwell
# -----------------------------------------------------------------------------
# INTEGRATION_TEST_01
# Basic LP, V=1, N=1, T=3
# -----------------------------------------------------------------------------

# --------------
# Imports
# --------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "Integration Test 1a"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":False, 
  "collisionAvoidance":False, 
  "plumeAvoidanceVehicle":False,
  "plumeAvoidanceObstacle":False,
  "finalConfigurationSelection":False}
# Define dynamics model
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
axLims = np.array([0, 1, -1, 1])

# ---------------
# Model Variables
# ---------------
N = 3    # Number of dimensions
T = 11   # Number of time steps
T_end = 100 # End time (s)
V = 1    # Number of vehicles
P = 10   # Plume Length (m)
W = 1    # Plume Width (m)
M = 1000 # Big M Method
m = np.array([1]) # Mass of satellites (kg)
minApproachDist = 1 # Min displacement between satellie and any obstacle in each dimension (m)
omega = 0 # (s) - only used in hills dynamics model

# Initial state vector
x_ini = np.array([[0, 0, 0, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[1, 0, 0, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 1*np.ones([V, N]) # thrust (N)
# Objects - only work in 2D+
objects = np.array([[]]) # [N1min, N2min, N1max, N2max, ...]
# Safety distance
r = 1*np.ones([N])