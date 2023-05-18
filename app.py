
# python C:\Program Files\IBM\ILOG\CPLEX_Studio1210\python\setup.py install
 
# TO DO:
# generate legends
# latex formatting of plots
# Implement fuel use? - currently optimizing for minimum total thrust (just a simple multiplication?)
# Add labels for x, y, z axis
# Calculate number of optimisation variables for each problem?

# Import function library
from lib.python.MILP_spacecraft_maneuvers import *

# Problem files
# Uncomment to run all Integration Test Problem Files
# testCases = ["INTEGRATION_TEST_01a",
#             "INTEGRATION_TEST_01b",
#             "INTEGRATION_TEST_01c",
#             "INTEGRATION_TEST_01d",
#             "INTEGRATION_TEST_02a",
#             "INTEGRATION_TEST_02b",
#             "INTEGRATION_TEST_03a",
#             "INTEGRATION_TEST_03b",
#             "INTEGRATION_TEST_04a",
#             "INTEGRATION_TEST_04b",
#             "INTEGRATION_TEST_05a",
#             "INTEGRATION_TEST_05b",
#             "INTEGRATION_TEST_06a",
#             "INTEGRATION_TEST_06b",
#             "INTEGRATION_TEST_05c",
#             "INTEGRATION_TEST_05d",
#             "INTEGRATION_TEST_06a",
#             "INTEGRATION_TEST_06b"]
# testCases = ["INTEGRATION_TEST_06a"]

# Uncomment to run all limitation test Problem Files
# testCases = ["LIMITATION_TEST_01",
#             "LIMITATION_TEST_02",
#             "LIMITATION_TEST_03",
#             "LIMITATION_TEST_04"]

# Uncomment to run all system test Problem Files
# testCases = [ 
#   "SYSTEM_TEST_06a",
#   "SYSTEM_TEST_06b",
#   "SYSTEM_TEST_06c",
#   "SYSTEM_TEST_06d",
#   "SYSTEM_TEST_06e"]

# Uncomment to run all paper Problem Files
testCases = [
  "PAPER_TEST_01a",
  "PAPER_TEST_01b",
  "PAPER_TEST_01c",
  "PAPER_03a",
  "PAPER_03b",
  "PAPER_04"]
testCases = [
  "PAPER_03a",
  "PAPER_03b"]
# Run models
for test in testCases:
  print("Test case: {}".format(test))
  file_path = "lib.python.tests." + test
  file = importlib.import_module(file_path)
  solution = optimizeTrajectory(
    file.activeConstraints, 
    file.N, 
    file.T, 
    file.T_end, 
    file.V, 
    file.P, 
    file.W, 
    file.M, 
    file.m, 
    file.minApproachDist, 
    file.x_ini, 
    file.x_fin, 
    file.x_lim, 
    file.u_lim, 
    file.objects, 
    file.r, 
    file.dynamicsModel,
    file.omega)
  objective = round(pulp.value(solution['model'].objective), 6)

  # Define outputs
  outputs = {
    "dataframe":True,
    "paths":True,
    "inputVector":True,
    "separation":False,
    "minSeparation":False,
    "saveFig":True}

  generateOutputs(outputs, file.title, test, solution['x'], solution['u'], file.objects, file.T, file.T_end, file.V, file.N, file.colourScheme, file.axLims, objective)

# Show plots
plt.show()