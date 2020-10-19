# Satellite equations of motion

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
# Palettes for categorical data visualisation
global paletteAccent6, paletteDark6 
paletteAccent6 = ["#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0", "#f0027f"]
paletteDark6 = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
import pulp
import importlib

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
# TO DO
# -----
# Determine satellite state (impinged, free, restricted)
# Implement tex in plots
# Automatically set legend positions
# -----------------------------------------------------------------------------

# --------
# Plotting
# --------

# Plot rectangle in figure. Allow choice of line style
def plotRectangle(axes, left, bottom, width, height, linestyle):
  linestyles = {
    "solid": '-',
    "dashed": '--',
    "dotted": ':',
  }
  linestyle = linestyles[linestyle]

  rect = patches.Rectangle((left, bottom), width, height, fill=None, alpha=1, linestyle=linestyle)
  axes.add_patch(rect)
  return

def plotDot(x, y, style):
  plt.plot(x, y, style)
  return

def plotPath(title, testCase, save, x, objects, T, V, N, colourMap):

  if N == 1: # Need to create function!
    fig = plot1DPath(title, testCase, save, x, objects, T, V, N, colourMap)
  elif N == 2:
    fig = plot2DPath(title, testCase, save, x, objects, T, V, N, colourMap)
  # elif N == 3:
  # else:
  return fig

# Plot Path of satellites in 2D space
def plot1DPath(title, testCase, save, x, objects, T, V, N, colourMap):
  
  # Variables
  title = append([title, " 1D Path"])
  fig = plt.figure(title)
  axes = plt.gca()
  d = np.empty([T, V, N])
  colourMap = globals()[colourMap]

  for i in range(T):
    for p in range(V):
      for n in range(N):
        d[i, p, n] = x[i, p, n].varValue
  for p in range(V):
    colour = colourMap[p]
    # Start and end positions
    plt.plot(d[0, p, 0], 0, 
      marker='+', 
      color=colour, 
      linestyle='None')
    plt.plot(d[T-1, p, 0], 0, 
      marker='x', 
      color=colour, 
      linestyle='None')
    # Satellite Path
    for i in range(1, T-1):
      plt.plot(d[i, p, 0], 0, 
        marker='o', 
        color=colour, 
        linestyle='None')

    # Plot layout
  axisRange = setAxis(N, d)
  #axes.set_xlim(axisRange[0,:])
  plt.legend()
  fig.suptitle(title, fontsize=12)

  # Save
  if(save):
    figName = append(["figures/", testCase, "_1DPath.png"])
    plt.savefig(figName, bbox_inches='tight')
  return fig

# Plot Path of satellites in 2D space
def plot2DPath(title, testCase, save, x, objects, T, V, N, colourMap):
  
  # Variables
  title = append([title, " 2D Path"])
  fig = plt.figure(title)
  axes = plt.gca()
  d = np.empty([T, V, N])
  colourMap = globals()[colourMap]

  for i in range(T):
    for p in range(V):
      for n in range(N):
        d[i, p, n] = x[i, p, n].varValue
  for p in range(V):
    colour = colourMap[p]
    # Start and end positions
    #label = "$x_{{(0, {}, :)}}$".format(p)
    plt.plot(d[0, p, 0], d[0, p, 1], 
      marker='+',
      color=colour,
      linestyle='None')
    #label = "$x_{{(T-1, {}, :)}}$".format(p)
    plt.plot(d[T-1, p, 0], d[T-1, p, 1], 
      marker='x', 
      color=colour, 
      linestyle='None')
    # Satellite Path
    label = "$x_{{(:, {}, :)}}$".format(p)
    cm = plt.cm.get_cmap('Reds')
    t = np.arange(T)
    plt.scatter(d[0:T, p, 0], d[0:T, p, 1] ,
      linewidth=2,
      marker = 'o',
      c=t,
      cmap=cm,
      label=label)
  if objects.shape[1] > 0:
    for i in range(objects.shape[0]):
      left   = objects[i, 0]
      bottom = objects[i, 1]
      width  = objects[i, 2] - objects[i, 0]
      height = objects[i, 3] - objects[i, 1]
      plotRectangle(axes, left, bottom, width, height, "solid") # bottom, left, 

  # Plot layout
  plt.grid(color='k', linestyle=':', alpha=0.5)
  axisRange = setAxis(N, d)
  axes.set_xlim(axisRange[0,:])
  axes.set_ylim(axisRange[1,:])
  plt.legend()
  fig.suptitle(title, fontsize=12)

  # Save
  if(save):
    figName = append(["figures/", testCase, "_2DPath.png"])
    plt.savefig(figName, bbox_inches='tight')
  return fig

# Plot thrusts of satellites
def plotThrust(title, testCase, save, u, T, V, N, colourMap):
  # TO DO:
  # Bar charts for each dimension, or sum fuel consumption?

  # Variables
  title = append([title, " thrust profile"])
  fig = plt.figure(title)
  thrust = np.empty([T, V, N])
  colourMap = globals()[colourMap]

  # Get thrust values
  for i in range(T):
    for p in range(V):
      for n in range(N):
        thrust[i, p, n] = u[i, p, n].varValue

  # Fuel consumption plots
  for n in range(N):
    for p in range(V):
      colour = colourMap[p]
      timeAx = np.linspace(0, T, num=T)
      axes = plt.subplot(N,1,n+1)
      axes.bar(timeAx, thrust[:, p, n], align='center', color=colour)
      plt.grid(color='k', linestyle=':', alpha=0.5)

  fig.suptitle(title, fontsize=12)

  # Create separate function for this
  if(save):
    figName = append(["figures/", testCase, "_plotThrust.png"])
    plt.savefig(figName, bbox_inches='tight')
  return fig

# Plot thrusts of satellites
def plotState(title, testCase, save, x, T, V, N, colourMap):

  # Variables
  title = append([title, " state profile"])
  fig = plt.figure(title)
  dis = np.empty([T, V, N])
  vel = np.empty([T, V, N])
  colourMap = globals()[colourMap]

  # Get thrust values
  for i in range(T):
    for p in range(V):
      for n in range(N):
        dis[i, p, n] = x[i, p, n].varValue
        vel[i, p, n] = x[i, p, n+N].varValue        

  # Fuel consumption plots
  for n in range(N):
    for p in range(V):
      colour = colourMap[p]
      timeAx = np.linspace(0, T, num=T)
      axes = plt.subplot(N,1,n+1)
      axes.bar(timeAx, thrust[:, p, n], align='center', color=colour)
      plt.grid(color='k', linestyle=':', alpha=0.5)

  fig.suptitle(title, fontsize=12)

  # Create separate function for this
  if(save):
    figName = append(["figures/", testCase, "_plotThrust.png"])
    plt.savefig(figName, bbox_inches='tight')
  return fig

# -------
# Outputs
# -------
# Print report of decision variables for optimized solution
def printOptimizationResults(title, model, x, u, T, V, N, pad, dp):
  if pad <= 5:
    print("ERROR: pad must be > 5")
    return
  if dp >= pad:
    print("ERROR: dp must be < pad")
    return

  padSpace = pad - 5

  print("-"*79)
  print(title)
  print("Status: {}".format(pulp.LpStatus[model.status]))
  print("Objective: {}".format(pulp.value(model.objective)))
  print("-"*79)

  if pulp.LpStatus[model.status] != "Infeasible":
    print("States")
    print("-"*79)  
    print("Timestep {:{pad}}| Vehicle {:{pad}}| Dimension {:{pad}}| Displacement{:{pad}}| Velocity{:{pad}}|".format("", "", "", "", "", pad=padSpace))
    for i in range(T):
      for p in range(V):
        for n in range(N):
          dis = x[i, p, n]
          dis_val = dis.varValue
          vel = x[i, p, N+n]
          vel_val = vel.varValue
          print("  {:{pad}}  |  {:{pad}}  |   {:{pad}}   |    {:{pad}.{dp}f}    |  {:{pad}.{dp}f}  |".format(i, p, n, dis_val, vel_val, pad=pad, dp=dp))
    print("-"*79)
    print("Inputs")
    print("-"*79)
    print("Timestep {:{pad}}| Vehicle {:{pad}}| Dimension {:{pad}}| Fuel        {:{pad}}|".format("", "", "", "", pad=padSpace))  
    for i in range(T):
      for p in range(V):
        for n in range(N):
          fuel = u[i, p, n]
          fuel_val = fuel.varValue
          print("  {:{pad}}  |  {:{pad}}  |   {:{pad}}   |    {:{pad}.{dp}f}    |".format(i, p, n, fuel_val, pad=pad, dp=dp))
    print("-"*79)
  return

# Store results in Pandas dataframe and export as csv
def exportCSV():
  return

# ------------------
# Optimization Model
# ------------------
# Return a list of constraints for model using free space dynamics
def freeSpaceDynamics(x, u, T, V, N, m, del_t):
  retVal = []
  for i in range(1, T): # First time step, no dynamics
    for p in range(V):
      for n in range(N):
        retVal.append(
          x[i, p, N+n] == x[i-1, p, N+n] + u[i-1, p, n]*1/m[p])
        retVal.append(
          x[i, p, n] == x[i-1, p, n] 
                      + x[i-1, p, N+n]*del_t 
                      + 0.5*u[i-1, p, n]*1/m[p]*del_t*del_t)
  return retVal

def linearHillsDynamics(x, u, T, V, N, m, del_t):
  retVal = []
  return retVal

# -------------
# Miscellaneous
# -------------
# Appends str2 to str1
def append(strArr):
  mainStr = strArr[0]
  if(len(strArr)) > 0:
    for i in range(1, len(strArr)):
      appendStr = str(strArr[i])
      for j in range(len(appendStr)):
        mainStr += appendStr[j]
  return mainStr

# Return axis limits for a dataset
# TO DO:
# - Include scale factor?
def setAxis(N, data):
  axisRange = np.empty([N, 2])
  bufferSpace = 0.5
  scaleFactor = 1.2
  for n in range(N):  
    axisRange[n, 0] = data[:, :, n].min() - bufferSpace
    axisRange[n, 1] = data[:, :, n].max() + bufferSpace
  return axisRange

def totalConstraints(T, V, N, numObjects):
  # Constraints up to obstacle avoidance (check!)
  numConstraints = 4*T*V*N + 2*(T-1)*V*N + 2*V*N + T*V*numObjects + T*V*N*numObjects
  return numConstraints

# -----------------------------------------------------------------------------
# Optimization Problem
# -----------------------------------------------------------------------------
# TO DO:
# ------
# - Hill's equations
# - Implement defined software architecture
# - Implement fuel use - currently optimizing for minimum total thrust
# -----------------------------------------------------------------------------

def optimizeTrajectory(N, T, T_end, V, P, W, M, m, TSFC, minApproachDist, x_ini, x_fin, x_lim, u_lim, objects, r):

  # --------------------
  # Calculated Variables
  # --------------------
  del_t = T_end/T
  numObjects = objects.shape[0]

  # ------------------
  # Decision variables
  # ------------------
  # Thrust
  u = pulp.LpVariable.dicts(
    "input", ((i, p, n) for i in range(T) for p in range(V) for n in range(N)),
    cat='Continuous')
  # Thrust Magnitude
  v = pulp.LpVariable.dicts(
    "inputMag", ((i, p, n) for i in range(T) for p in range(V) for n in range(N)),
    lowBound=0,
    cat='Continuous')
  # State
  x = pulp.LpVariable.dicts(
    "state", ((i, p, k) for i in range(T) for p in range(V) for k in range(2*N)),
    cat='Continuous')
  # Object collision
  a = pulp.LpVariable.dicts(
    "objCol", ((i, p, l, k) for i in range(T) for p in range(V) for l in range(numObjects) for k in range(2*N)),
    cat='Binary')
  # Satellite Collision Avoidance
  b = pulp.LpVariable.dicts(
    "satelliteCollisionAvoidance", ((i, p, q, k) for i in range(T) for p in range(V) for q in range(V) for k in range(2*N)),
    cat='Binary')
  # Plume Impingement
  c_plus = pulp.LpVariable.dicts(
    "plumeImpingementPositive", ((i, p, q, n, k) for i in range(T) for p in range(V) for q in range(V) for n in range(N) for k in range(2*N)),
    cat='Binary')
  c_minus = pulp.LpVariable.dicts(
    "plumeImpingementNegative", ((i, p, q, n, k) for i in range(T) for p in range(V) for q in range(V) for n in range(N) for k in range(2*N)),
    cat='Binary')

  # ------------------
  # Optimization Model
  # ------------------

  # Instantiate Model
  model = pulp.LpProblem("Satellite Fuel Minimization Problem", pulp.LpMinimize)
  # Objective Function
  model += pulp.lpSum(v[i, p, n] for i in range(T) for p in range(V) for n in range(N)), "Fuel Minimization"

  # -----------
  # Constraints
  # -----------

  # Basic Constraints
  # -----------------
  # Constrain thrust magnitude to abs(u[i, p, n])
  for i in range(T):
    for p in range(V):
      for n in range(N):
        model +=  u[i, p, n] <= v[i, p, n]
        model += -u[i, p, n] <= v[i, p, n]
  # State and Input vector start and end values
  for p in range(V):
    for k in range(2*N):
      model += x[0, p, k]   == x_ini[p, k]
      model += x[T-1, p, k] == x_fin[p, k]
  # Model Dynamics
  for constraint in freeSpaceDynamics(x, u, T, V, N, m, del_t):
    model += constraint
  # State and Input vector limits
  for i in range(T):
    for p in range(V):
      for n in range(N):
        model += u[i, p, n] <=  u_lim[p, n]
        model += u[i, p, n] >= -u_lim[p, n]
      for k in range(2*N): # Necessary?
        model += x[i, p, k] <=  x_lim[p, k]
        model += x[i, p, k] >= -x_lim[p, k]

  # Obstacle Avoidance
  # ------------------
  if objects.shape[1] > 0:
    for i in range(T):
      for p in range(V):
        for l in range(numObjects):
          model += pulp.lpSum(a[i, p, l, n] for n in range(2*N)) <= 2*N-1
          for n in range(N):
            model += x[i, p, n] >= objects[l, N+n] + minApproachDist - M*a[i, p, l, N+n]
            model += x[i, p, n] <= objects[l, n] - minApproachDist + M*a[i, p, l, n]

  # Collision Avoidance
  # -------------------
  if V > 1: # If more than one vehicle
      for i in range(T):
        for p in range(V):
          for q in range(V):
            if q > p:
              model += pulp.lpSum(b[i, p, q, k] for k in range(2*N)) <= 2*N-1
              for n in range(N):
                model += x[i, p, n] - x[i, q, n] >= r[n] - M*b[i, p, q, n]
                model += x[i, q, n] - x[i, p, n] >= r[n] - M*b[i, p, q, n+N]

  # Plume Impingement
  # -----------------
  # Positive thrust
  if V > 1: # If more than one vehicle
      for i in range(T):
        for p in range(V):
          for q in range(V):
            if q != p:
              for n in range(N):
                model += pulp.lpSum(c_plus[i, p, q, n, k] for k in range(2*N)) <= 2*N
                model += -u[i, p, n] >= - M*c_plus[i, p, q, n, 0]
                model += x[i, p, n] - x[i, q, n] >= P - M*c_plus[i, p, q, n, n]
                model += x[i, q, n] - x[i, p, n] >= - M*c_plus[i, p, q, n, n+N]
                for m in range(N):
                  if m != n:
                    x[i, p, m] - x[i, q, m] >= W - M*c_plus[i, p, q, n, m]
                    x[i, q, m] - x[i, p, m] >= W - M*c_plus[i, p, q, n, m+N]
  # Negative thrust
      for i in range(T):
        for p in range(V):
          for q in range(V):
            if q != p:
              for n in range(N):
                model += pulp.lpSum(c_minus[i, p, q, n, k] for k in range(2*N)) <= 2*N
                model += u[i, p, n] >= - M*c_minus[i, p, q, n, 0]
                model += x[i, p, n] - x[i, q, n] >= - M*c_minus[i, p, q, n, n]
                model += x[i, q, n] - x[i, p, n] >= P - M*c_minus[i, p, q, n, n+N]
                for m in range(N):
                  if m != n:
                    x[i, p, m] - x[i, q, n] >= W - M*c_minus[i, p, q, n, m]
                    x[i, q, m] - x[i, p, m] >= W - M*c_minus[i, p, q, n, m+N]


  # Plume Avoidance for Vehicles
  # ----------------------------

  # Plume Avoidance for Obstacles
  # -----------------------------

  # Final Configuration Selection
  # -----------------------------

  # Solve model and return results in dictionary
  model.solve(pulp.CPLEX())

  # Create Pandas dataframe for results and return

  return {'model':model, 'x':x, 'u':u}

# --------------
# Run test cases
# --------------
testCases = ["BUGFIX03"]

for test in testCases:
  file = importlib.import_module(test)
  solution = optimizeTrajectory(file.N, file.T, file.T_end, file.V, file.P, file.W, file.M, file.m, file.TSFC, file.minApproachDist, file.x_ini, file.x_fin, file.x_lim, file.u_lim, file.objects, file.r)

  printOptimizationResults(file.title, solution['model'], solution['x'], solution['u'], file.T, file.V, file.N, 8, 3)
  fig = plotPath(file.title, test, False, solution['x'], file.objects, file.T, file.V, file.N, "paletteDark6")
  fig = plotThrust(file.title, test, False, solution['u'], file.T, file.V, file.N, "paletteDark6")

plt.show()