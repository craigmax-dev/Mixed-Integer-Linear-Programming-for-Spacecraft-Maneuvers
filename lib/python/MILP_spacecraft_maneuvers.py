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
import importlib
import math as mt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pulp
import pandas as pd
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Latex formatting
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

# --------
# Plotting
# --------

# Plot rectangle in figure
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

# Plot cube in figure
def plotCube(axes, obj, linestyle):
  linestyles = {
    "solid": '-',
    "dashed": '--',
    "dotted": ':',
  }
  for s, e in combinations(np.array(list(product(obj[0:2], obj[2:4], obj[4:6]))), 2):
    if (np.sum(np.abs(s-e)) == obj[1]-obj[0] or np.sum(np.abs(s-e)) == obj[3]-obj[2] or np.sum(np.abs(s-e)) == obj[5]-obj[4]):
      axes.plot3D(*zip(s, e), color='k', linestyle=linestyle)
  return

# Plot satellite paths
def plotPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective):
  if N ==1:
    plot1DPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective)
  elif N == 2: 
    plot2DPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective)
  elif N == 3:
    plot3DPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective)
  return

# Plot Path of satellites in 2D space
def plot2DPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective):

  title = "2D path, $J = {}$".format(objective)
  
  # Variables
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
    plt.plot(d[0, p, 0], d[0, p, 1], 
      marker='+',
      markersize=10,
      mew=2,
      color=colour,
      linestyle='None')
    plt.plot(d[T-1, p, 0], d[T-1, p, 1], 
      marker='x',
      markersize=10, 
      mew=2,
      color=colour, 
      linestyle='None')
    # Satellite Path
    label = "Vehicle {}".format(p+1)
    # cm = plt.cm.get_cmap('Reds')
    # t = np.arange(T)
    plt.plot(d[:, p, 0], d[:, p, 1] ,
      linewidth=1,
      color=colour,
      label=label)
    plt.plot(d[1:T-1, p, 0], d[1:T-1, p, 1] ,
      marker='.',
      markersize=10, 
      color=colour,
      label='_Hidden')
  if objects.shape[1] > 0:
    for i in range(objects.shape[0]):
      left   = objects[i, 0]
      bottom = objects[i, 2]
      width  = objects[i, 1] - objects[i, 0]
      height = objects[i, 3] - objects[i, 2]
      plotRectangle(axes, left, bottom, width, height, "solid")

  # Plot layout
  plt.grid(color='k', linestyle=':', alpha=0.5)
  axisRange = setAxis(N, d)

  axes.set_xlim(axLims[0] - 1, axLims[1] + 1)
  axes.set_ylim(axLims[2] - 1, axLims[3] + 1)
  plt.xlabel('$x (m)$')
  plt.ylabel('$y (m)$')  
  plt.legend()
  fig.suptitle(title, fontsize=12)

  # Save
  if(save):
    figName = append(["figures/", testCase, "_2DPath.png"])
    plt.savefig(figName, bbox_inches='tight')
  return

def plot3DPath(title, testCase, save, x, objects, T, V, N, colourMap, axLims, objective):

  title = "3D path, $J = {}$".format(objective)

  # Variables
  fig = plt.figure(title)
  axes = fig.add_subplot(111, projection='3d')
  colourMap = globals()[colourMap]

  d = np.empty([T, V, N])
  for i in range(T):
    for p in range(V):
      for n in range(N):
        d[i, p, n] = x[i, p, n].varValue
  for p in range(V):
    colour = colourMap[p]
    # Satellite Path
    label = "Vehicle {}".format(p+1)
    # cm = plt.cm.get_cmap('Reds')
    # t = np.arange(T)
    # Start and end positions
    plt.plot(d[0, p, 0], d[0, p, 1] , d[0, p, 2], 
      marker='+',
      markersize=5,
      color=colour,
      linestyle='None')
    plt.plot(d[T-1, p, 0], d[T-1, p, 1] , d[T-1, p, 2], 
      marker='x',
      markersize=5, 
      color=colour, 
      linestyle='None')
    plt.plot(d[:, p, 0], d[:, p, 1] , d[:, p, 2],  
      linewidth=2,
      color=colour,
      label=label)
    plt.plot(d[1:T-1, p, 0], d[1:T-1, p, 1] , d[1:T-1, p, 2],  
      marker='.',
      markersize=5, 
      color=colour,
      label='_Hidden')
  if objects.shape[1] > 0:
    for l in range(objects.shape[0]):
      obj = objects[l, :]
      plotCube(axes, obj, 'dashed')

  # Plot layout
  plt.grid(color='k', linestyle=':', alpha=0.5)
  # axisRange = setAxis(N, d)
  # axes.set_aspect('equal')
  axes.set_xlim3d(axLims[0] - 1, axLims[1] + 1)
  axes.set_ylim3d(axLims[2] - 1, axLims[3] + 1)
  axes.set_zlim3d(axLims[4] - 1, axLims[5] + 1)
  plt.xlabel('$x (m)$')
  plt.ylabel('$y (m)$')  
  plt.ylabel('$z (m)$')  
  plt.legend()
  fig.suptitle(title, fontsize=12)

  # Save
  if(save):
    figName = append(["figures/", testCase, "_3DPath.png"])
    plt.savefig(figName, bbox_inches='tight')
  return

# Plot thrusts of satellites
def plotThrust(title, testCase, save, u, T, T_end, V, N, colourMap, objective):

  # Variables
  title = "Thrust Profile, $J = {}$".format(objective)

  fig = plt.figure(title)
  thrust = np.zeros([T+1, V, N])
  colourMap = globals()[colourMap]

  # Get thrust values
  for i in range(T):
    for p in range(V):
      for n in range(N):
        if(i != T-1):
          thrust[i+1, p, n] = u[i, p, n].varValue  

  # Fuel consumption plots
  axesStr = ["x", "y", "z"]
  for n in range(N):
    for p in range(V):
      label = "Vehicle {}".format(p)
      colour = colourMap[p]
      timeAx = np.linspace(0, T_end+T_end/(T-1), num=T+1)
      axes = plt.subplot(N,1,n+1)
      axes.step(timeAx, thrust[:, p, n], color=colour, label=label)
      plt.grid(color='k', linestyle=':', alpha=0.5)
    plt.ylabel('$u_{} (N)$'.format(axesStr[n]))  
  plt.xlabel('$t (s)$')


  fig.suptitle(title, fontsize=12)

  # Create separate function for this
  if(save):
    figName = append(["figures/", testCase, "_thrust.png"])
    plt.savefig(figName, bbox_inches='tight')
  return

# Plot satellite separation
def plotSeparation(title, testCase, save, x, T, T_end, V, N, colourMap):

  # Variables
  title = "Separation"


  fig = plt.figure(title)
  if V <= 1:
    return fig
  numCombinations = int(mt.factorial(V) / (2 * mt.factorial(V - 2)))
  separation = np.zeros([numCombinations, T, N])
  colourMap = globals()[colourMap]

  # Get separations
  current = -1
  for p in range(V):
    for q in range(V):
      if q > p:
        current += 1
        for i in range(T):
          for n in range(N):
            separation[current, i, n] = abs(x[i, p, n].varValue - x[i, q, n].varValue)

  # Separation plots
  for n in range(N):
    for combo in range(numCombinations):
      colour = colourMap[combo]
      timeAx = np.linspace(0, T_end, num=T)
      axes = plt.subplot(N,1,n+1)
      axes.plot(timeAx, separation[combo, :, n], color=colour)
      plt.grid(color='k', linestyle=':', alpha=0.5)

  fig.suptitle(title, fontsize=12)

  # Create separate function for this
  if(save):
    figName = append(["figures/", testCase, "_separation.png"])
    plt.savefig(figName, bbox_inches='tight')
  return

# Plot minimum satellite separation in each axis
def plotMinSeparation(title, testCase, save, x, T, T_end, V, N, colourMap):

  # Variables
  title = "Minimum Separation"

  fig = plt.figure(title)
  if V <= 1:
    return fig
  numCombinations = int(mt.factorial(V) / (2 * mt.factorial(V - 2)))
  separation = np.zeros([numCombinations, T, N])
  minSeparation = np.zeros([numCombinations, T])
  colourMap = globals()[colourMap]

  # Get separations
  current = -1
  for p in range(V):
    for q in range(V):
      if q > p:
        current += 1
        for i in range(T):
          for n in range(N):
            separation[current, i, n] = abs(x[i, p, n].varValue - x[i, q, n].varValue)

  for combo in range(numCombinations):
    for i in range(T):
      minSeparation[combo, i] = max(separation[combo, i, :])
      
  # Separation plot
  for combo in range(numCombinations):
    colour = colourMap[combo]
    timeAx = np.linspace(0, T_end, num=T)
    plt.plot(timeAx, minSeparation[combo, :], color=colour)
    plt.grid(color='k', linestyle=':', alpha=0.5)

  plt.axhline(y=1, xmin=0, xmax=T_end, linewidth=2, color = 'k', linestyle = '--')
  fig.suptitle(title, fontsize=12)
  plt.ylim(0, np.amax(minSeparation))

  # Create separate function for this
  if(save):
    figName = append(["figures/", testCase, "_minSeparation.png"])
    plt.savefig(figName, bbox_inches='tight')
  return

# -------
# Outputs
# -------
# Generate outputs for a problem file
def generateOutputs(outputs, title, test, x, u, objects, T, T_end, V, N, colourScheme, axLims, objective):
  if outputs["dataframe"]:
    exportDataframe(title, test, x, u)
  if outputs["paths"]:
    plotPath(title, test, outputs["saveFig"], x, objects, T, V, N, colourScheme, axLims, objective)
  if outputs["inputVector"]:
    plotThrust(title, test, outputs["saveFig"], u, T, T_end, V, N, colourScheme, objective)
  if outputs["separation"]:
    plotSeparation(title, test, outputs["saveFig"], x, T, T_end, V, N, colourScheme)
  if outputs["minSeparation"]:
    plotMinSeparation(title, test, outputs["saveFig"], x, T, T_end, V, N, colourScheme)
  return

def exportDataframe(title, test, x, u):
  # Set up DataFrame for x
  x_df = pd.DataFrame.from_dict(
    x, 
    orient="index", 
    columns = ["variable_object"])
  x_df["solution_value"] = x_df["variable_object"].apply(lambda 
    item: item.varValue)
  x_df.drop(
    columns=["variable_object"], 
    inplace=True)
  # Set up DataFrame for u
  u_df = pd.DataFrame.from_dict(
    u, 
    orient="index", 
    columns = ["variable_object"])
  u_df["solution_value"] = u_df["variable_object"].apply(lambda 
    item: item.varValue)
  u_df.drop(
    columns=["variable_object"], 
    inplace=True)

  # Export
  x_df.to_csv(append(["data/OPT_SOL_", test, "_x.csv"]))
  u_df.to_csv(append(["data/OPT_SOL_", test, "_u.csv"]))
  return

# ------------------
# Optimization Model
# ------------------
# Return a list of constraints for model using free space dynamics
def freeSpaceDynamics(x, u, T, V, N, m, del_t):
  retVal = []
  for i in range(1, T):
    for p in range(V):
      for n in range(N):
        # Velocities
        retVal.append(
          x[i, p, N+n] == x[i-1, p, N+n] + u[i-1, p, n]*1/m[p]*del_t)
        # Displacements
        retVal.append(
          x[i, p, n] == x[i-1, p, n] 
                      + x[i-1, p, N+n]*del_t 
                      + 0.5*u[i-1, p, n]*1/m[p]*del_t*del_t)
  return retVal

def linearHillsDynamics(x, u, T, V, N, m, del_t, omega):
# Return a list of constraints for model using free space dynamics
# NOTE: not functional - results in no feasible solution
  retVal = []
  for i in range(1, T):
    for p in range(V):
      retVal.append(
        x[i, p, 0] == x[i-1, p, 0] + x[i-1, p, N]*del_t + 0.5*(x[i-1, p, 0]*3*omega*omega + x[i-1, p, 1+N]*2*omega + u[i-1, p, 0]*1/m[p])*del_t*del_t)
      retVal.append(
        x[i, p, 1] == x[i-1, p, 1] + x[i-1, p, N+1]*del_t + 0.5*(-x[i-1, p, N]*2*omega + u[i-1, p, 1]*1/m[p])*del_t*del_t)
      retVal.append(
        x[i, p, 2] == x[i-1, p, 2] + x[i-1, p, N+2]*del_t + 0.5*(-x[i-1, p, 2]*omega*omega + u[i-1, p, 2]*1/m[p])*del_t*del_t)
      retVal.append(
        x[i, p, N] == x[i-1, p, N] + (x[i-1, p, 0]*3*omega*omega + x[i-1, p, 1+N]*2*omega + u[i-1, p, 0]*1/m[p])*del_t)
      retVal.append(
        x[i, p, N+1] == x[i-1, p, N+1] + (-x[i-1, p, N]*2*omega + u[i-1, p, 1]*1/m[p])*del_t)
      retVal.append(
        x[i, p, N+2] == x[i-1, p, N+2] + (-x[i-1, p, 2]*omega*omega + u[i-1, p, 2]*1/m[p])*del_t)
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
def setAxis(N, data):
  axisRange = np.empty([N, 2])
  bufferSpace = 0.5
  scaleFactor = 1.2
  for n in range(N):  
    axisRange[n, 0] = data[:, :, n].min() - bufferSpace
    axisRange[n, 1] = data[:, :, n].max() + bufferSpace
  return axisRange

def totalConstraints(T, V, N, numObjects):
  # Constraints up to obstacle avoidance
  numConstraints = 4*T*V*N + 2*(T-1)*V*N + 2*V*N + T*V*numObjects + T*V*N*numObjects
  return numConstraints

def optimizeTrajectory(activeConstraints, 
  N, T, T_end, V, P, W, M, m, minApproachDist, 
  x_ini, x_fin, x_lim, u_lim, objects, r, dynamicsModel, omega):

  # --------------------
  # Calculated Variables
  # --------------------
  del_t = T_end/(T-1)
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
      if(activeConstraints["finalConfigurationSelection"] == False):
        model += x[T-1, p, k] == x_fin[p, k]
  # Model Dynamics
  if dynamicsModel == "freeSpace":
    for constraint in freeSpaceDynamics(x, u, T, V, N, m, del_t):
      model += constraint
  elif dynamicsModel == "hills":
    for constraint in linearHillsDynamics(x, u, T, V, N, m, del_t, omega):
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
  if (activeConstraints["obstacleAvoidance"] == True and objects.shape[1] > 0):
    #Obstacle avoidance binary decision variable
    a = pulp.LpVariable.dicts(
      "objCol", ((i, p, l, k) for i in range(T) for p in range(V) for l in range(numObjects) for k in range(2*N)),
      cat='Binary')
    for i in range(T):
      for p in range(V):
        for l in range(objects.shape[0]):
          model += pulp.lpSum(a[i, p, l, n] for n in range(2*N)) <= 2*N-1
          for n in range(N):
            model += x[i, p, n] >= objects[l, n*2+1] + minApproachDist - M*a[i, p, l, N+n]           
            model += x[i, p, n] <= objects[l, n*2] - minApproachDist + M*a[i, p, l, n]

  # Collision Avoidance
  # -------------------
  if (activeConstraints["collisionAvoidance"] == True and V > 1):
    # Collision avoidance binary decision variable
    b = pulp.LpVariable.dicts(
    "satelliteCollisionAvoidance", ((i, p, q, k) for i in range(T) for p in range(V) for q in range(V) for k in range(2*N)),
    cat='Binary')
    for i in range(T):
      for p in range(V):
        for q in range(V):
          if q > p:
            model += pulp.lpSum(b[i, p, q, k] for k in range(2*N)) <= 2*N-1
            for n in range(N):
              model += x[i, p, n] - x[i, q, n] >= r[n] - M*b[i, p, q, n]
              model += x[i, q, n] - x[i, p, n] >= r[n] - M*b[i, p, q, n+N]

  # Plume Avoidance for Vehicles
  # ----------------------------
  if (activeConstraints["plumeAvoidanceVehicle"] == True):
    # Positive thrust
    c_pos = pulp.LpVariable.dicts(
    "plumeAvoidanceVehiclePos", ((i, p, q, n, k) for i in range(T) for p in range(V) for q in range(V) for n in range(N) for k in range(2*N+1)),
    cat='Binary')
    for i in range(T):
      for p in range(V):
        for q in range(V):
          if q != p:
            for n in range(N):
              model += pulp.lpSum(c_pos[i, p, q, n, k] for k in range(2*N+1)) <= 2*N
              model += -u[i, p, n] >= - M*c_pos[i, p, q, n, 0]
              model += x[i, p, n] - x[i, q, n] >= P - M*c_pos[i, p, q, n, n+1]
              model += x[i, q, n] - x[i, p, n] >= - M*c_pos[i, p, q, n, n+N+1]
              for m in range(N):
                if m != n:
                  model += x[i, p, m] - x[i, q, m] >= W - M*c_pos[i, p, q, n, m+1]
                  model += x[i, q, m] - x[i, p, m] >= W - M*c_pos[i, p, q, n, m+N+1]
    c_neg = pulp.LpVariable.dicts(
    "plumeAvoidanceVehicleNeg", ((i, p, q, n, k) for i in range(T) for p in range(V) for q in range(V) for n in range(N) for k in range(2*N+1)),
    cat='Binary')
    for i in range(T):
      for p in range(V):
        for l in range(numObjects):
          for n in range(N):
            model += pulp.lpSum(c_neg[i, p, q, n, k] for k in range(2*N+1)) <= 2*N
            model += u[i, p, n] >= - M*c_neg[i, p, q, n, 0]
            model += x[i, p, n] - x[i, q, n] >= - M*c_neg[i, p, q, n, n+1]
            model += x[i, q, n] - x[i, p, n] >= P - M*c_neg[i, p, q, n, n+N+1]
            for m in range(N):
              if m != n:
                model += x[i, p, m] - x[i, q, m] >= W - M*c_neg[i, p, q, n, m+1]
                model += x[i, q, m] - x[i, p, m] >= W - M*c_neg[i, p, q, n, m+N+1]

  # Plume Avoidance for Obstacles
  # -----------------------------
  if (activeConstraints["plumeAvoidanceObstacle"] == True and objects.shape[1] > 0):
    # Positive thrust
    d_pos = pulp.LpVariable.dicts(
    "plumeAvoidanceObstaclePos", ((i, p, l, n, k) for i in range(T) for p in range(V) for l in range(numObjects) for n in range(N) for k in range(2*N+1)),
    cat='Binary')
    for i in range(T):
      for p in range(V):
        for l in range(numObjects):
          for n in range(N):
            model += pulp.lpSum(d_pos[i, p, l, n, k] for k in range(2*N+1)) <= 2*N
            model += -u[i, p, n] >= - M*d_pos[i, p, l, n, 0]
            model += x[i, p, n] - objects[l, n*2+1] >= P - M*d_pos[i, p, l, n, n+1]
            model += objects[l, n*2] - x[i, p, n]   >= - M*d_pos[i, p, l, n, n+N+1]
            for m in range(N):
              if m != n:
                model += x[i, p, m] - objects[l, m*2+1] >= W - M*d_pos[i, p, l, n, m+1]
                model += objects[l, m*2] - x[i, p, m]   >= W - M*d_pos[i, p, l, n, m+N+1]
    d_neg = pulp.LpVariable.dicts(
    "plumeAvoidanceObstacleNeg", ((i, p, l, n, k) for i in range(T) for p in range(V) for l in range(numObjects) for n in range(N) for k in range(2*N+1)),
    cat='Binary')
    for i in range(T):
      for p in range(V):
        for l in range(numObjects):
          for n in range(N):
            model += pulp.lpSum(d_neg[i, p, l, n, k] for k in range(2*N+1)) <= 2*N
            model += u[i, p, n] >= - M*d_neg[i, p, l, n, 0]
            model += x[i, p, n] - objects[l, n*2+1] >= - M*d_neg[i, p, l, n, n+1]
            model += objects[l, n*2] - x[i, p, n]   >= P - M*d_neg[i, p, l, n, n+N+1]
            for m in range(N):
              if m != n:
                model += x[i, p, m] - objects[l, m*2+1] >= W - M*d_neg[i, p, l, n, m+1]
                model += objects[l, m*2] - x[i, p, m]   >= W - M*d_neg[i, p, l, n, m+N+1]

  # Final Configuration Selection
  # -----------------------------
  if (activeConstraints["finalConfigurationSelection"]):
    G = V
    f = pulp.LpVariable.dicts(
    "finalConfigurationSelection", ((p, g, r) for p in range(V) for g in range(G) for r in range(V)),
    cat='Binary')

    for p in range(V):
      for k in range(2*N):
        model += x[T-1, p, k] == pulp.lpSum(x_fin[r, k]*f[p, g, r] for g in range(G) for r in range(V))

    # model += x[T-1, p, k] == x_fin[p, k]

    for p in range(V):
      model += pulp.lpSum(f[p, g, r] for g in range(G) for r in range(V)) == 1

    for p in range(V):
      for g in range(G):
        model += pulp.lpSum(f[p, g, r] for r in range(V)) == pulp.lpSum(f[r, g, p] for r in range(V))

    for g in range(G):
      model += pulp.lpSum(f[p, g, r] for p in range(V) for r in range(V)) == V*pulp.lpSum(f[0, g, r] for r in range(V))
    

  # Solve model and return results in dictionary
  model.solve(pulp.CPLEX())

  # Create Pandas dataframe for results and return
  return {'model':model, 'x':x, 'u':u}