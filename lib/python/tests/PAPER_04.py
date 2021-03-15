# -----------------------------------------------------------------------------
# PAPER_04
# -----------------------------------------------------------------------------
# Satellite Formation Reconfiguration

  # TO DO:
# Find thrust for satellites.

# --------------
# Imports
# --------------
import numpy as np

# --------------
# Model Settings
# --------------
# Problem File title
title = "Paper Simulation 4: Satellite Formation Reconfiguration"
# Define active constraints
activeConstraints = { 
  "basic":True, 
  "obstacleAvoidance":True, 
  "collisionAvoidance":True, 
  "plumeAvoidanceVehicle":True,
  "plumeAvoidanceObstacle":True,
  "finalConfigurationSelection":True}
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
axLims = np.array([-60, 60, -50, 50, -10, 10])
# intrack, radial, out of plane

# ---------------
# Model Variables
# ---------------
N = 3       # Number of dimensions
T = 25     # Number of time steps
T_end = 1000 # End time (s)
V = 3       # Number of vehicles
P = 100      # Plume Length (m)
W = 10       # Plume Width (m)
M = 1000000 # Big M Method
m = np.array([50, 50, 50])  # Mass of satellites (kg)
minApproachDist = 1   # Min displacement between satellie and any obstacle in each dimension (m)
omega = 0 # (s)
dynamicsModel = "hills" # Define dynamics model - "hills" or "freeSpace"

# Initial state vector
x_ini = np.array([[-2, 12, 0, 0, 0, 0]]) # x_i = 0, v_i = 0
# Final state vector
x_fin = np.array([[3, 17, 4, 0, 0, 0]]) # x_i = 1, v_i = 0
# State vector limits
x_lim = 100*np.ones([V, 2*N])
# Input vector limits
u_lim = 1e-6*np.ones([V, N]) # Thrust (N)
# Objects - only work in 2D+
objects = np.array([
  [0, 5, 0, 30, 0, 5], 
  [0, 20, 14, 15, 2, 3], 
  [-10, 0, 13, 16, 1, 4], 
  [2, 3, 5, 10, -10, 10]]) #
# Safety distance
r = 1*np.ones([N])