# -----
# TO DO
# -----
# Label all axes of plotted figures with variable and units

# Import function library
from MILP_LIB import *

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

# Uncomment to run all System Test Problem Files
# testCases = ["SYSTEM_TEST_01a",
# 			"SYSTEM_TEST_01b",
# 			"SYSTEM_TEST_01c"]

# Uncomment to run all limitation test Problem Files
# testCases = ["LIMITATION_TEST_01",
#             "LIMITATION_TEST_02",
#             "LIMITATION_TEST_03, # Plotting time points would be better
#             "LIMITATION_TEST_04,
#             "LIMITATION_TEST_05"] # Is there one for configuration selection?

testCases = ["SYSTEM_TEST_01a",
			"SYSTEM_TEST_01b",
			"SYSTEM_TEST_01c"]

# Run models
for test in testCases:
  print("Test case: {}".format(test))
  file = importlib.import_module(test)
  solution = optimizeTrajectory(file.activeConstraints, file.N, file.T, file.T_end, file.V, file.P, file.W, file.M, file.m, file.minApproachDist, file.x_ini, file.x_fin, file.x_lim, file.u_lim, file.objects, file.r)

  objective = round(pulp.value(solution['model'].objective), 6)

  generateOutputs(file.outputs, file.title, test, solution['x'], solution['u'], file.objects, file.T, file.T_end, file.V, file.N, file.colourScheme, file.axLims, objective)

# Show plots
plt.show()