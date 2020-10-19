import cplex
import sys

def sample1(filename):
	c = cplex.Cplex(filename)

	try:
		c.solve()
	except CplexSolverError:
		print("Exception raised during solve")
		return

	status = c.solution.get_status()
	print("Solution status = {}:{}".format(status, c.solution.status[status]))
	print("Objective value = {}".format(c.solution.get_objective_value()))
	