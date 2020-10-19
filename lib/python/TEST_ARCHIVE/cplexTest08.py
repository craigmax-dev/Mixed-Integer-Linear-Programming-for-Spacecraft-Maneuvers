from sys import stdout
import docplex.cp

url = None
key = None

from docplex.cp.model import *
# List of possible truck configurations. Each tuple is (load, cost) with:
#    load: max truck load for this configuration,
#    cost: cost for loading the truck in this configuration
TRUCK_CONFIGURATIONS = ((11, 2), (11, 2), (11, 2), (11, 3), (10, 3), (10, 3), (10, 4))

# List of customer orders.
# Each tuple is (customer index, volume, product type)
CUSTOMER_ORDERS = ((0, 3, 1), (0, 4, 2), (0, 3, 0), (0, 2, 1), (0, 5, 1), (0, 4, 1), (0, 11, 0),
                   (1, 4, 0), (1, 5, 0), (1, 2, 0), (1, 4, 2), (1, 7, 2), (1, 3, 2), (1, 5, 0), (1, 2, 2),
                   (2, 5, 1), (2, 6, 0), (2, 11, 2), (2, 1, 0), (2, 6, 0), (2, 3, 0))

# Transition costs between configurations.
# Tuple (A, B, TCost) means that the cost of  modifying the truck from configuration A to configuration B is TCost
CONFIGURATION_TRANSITION_COST = tuple_set(((0, 0,  0), (0, 1,  0), (0, 2,  0), (0, 3, 10), (0, 4, 10),
                                           (0, 5, 10), (0, 6, 15), (1, 0,  0), (1, 1,  0), (1, 2,  0),
                                           (1, 3, 10), (1, 4, 10), (1, 5, 10), (1, 6, 15), (2, 0,  0),
                                           (2, 1,  0), (2, 2,  0), (2, 3, 10), (2, 4, 10), (2, 5, 10),
                                           (2, 6, 15), (3, 0,  3), (3, 1,  3), (3, 2,  3), (3, 3,  0),
                                           (3, 4, 10), (3, 5, 10), (3, 6, 15), (4, 0,  3), (4, 1,  3),
                                           (4, 2,  3), (4, 3, 10), (4, 4,  0), (4, 5, 10), (4, 6, 15),
                                           (5, 0,  3), (5, 1,  3), (5, 2,  3), (5, 3, 10), (5, 4, 10),
                                           (5, 5,  0), (5, 6, 15), (6, 0,  3), (6, 1,  3), (6, 2,  3),
                                           (6, 3, 10), (6, 4, 10), (6, 5, 10), (6, 6,  0)
                                           ))

# Compatibility between the product types and the configuration of the truck
# allowedContainerConfigs[i] = the array of all the configurations that accept products of type i
ALLOWED_CONTAINER_CONFIGS = ((0, 3, 4, 6),
                             (1, 3, 5, 6),
                             (2, 4, 5, 6))

nbTruckConfigs = len(TRUCK_CONFIGURATIONS)
maxTruckConfigLoad = [tc[0] for tc in TRUCK_CONFIGURATIONS]
truckCost = [tc[1] for tc in TRUCK_CONFIGURATIONS]
maxLoad = max(maxTruckConfigLoad)

nbOrders = len(CUSTOMER_ORDERS)
nbCustomers = 1 + max(co[0] for co in CUSTOMER_ORDERS)
volumes = [co[1] for co in CUSTOMER_ORDERS]
productType = [co[2] for co in CUSTOMER_ORDERS]

# Max number of truck deliveries (estimated upper bound, to be increased if no solution)
maxDeliveries = 15

mdl = CpoModel(name="trucks")


# Configuration of the truck for each delivery
truckConfigs = integer_var_list(maxDeliveries, 0, nbTruckConfigs - 1, "truckConfigs")
# In which delivery is an order
where = integer_var_list(nbOrders, 0, maxDeliveries - 1, "where")
# Load of a truck
load = integer_var_list(maxDeliveries, 0, maxLoad, "load")
# Number of deliveries that are required
nbDeliveries = integer_var(0, maxDeliveries)
# Identification of which customer is assigned to a delivery
customerOfDelivery = integer_var_list(maxDeliveries, 0, nbCustomers, "customerOfTruck")
# Transition cost for each delivery
transitionCost = integer_var_list(maxDeliveries - 1, 0, 1000, "transitionCost")

# transitionCost[i] = transition cost between configurations i and i+1
for i in range(1, maxDeliveries):
    auxVars = (truckConfigs[i - 1], truckConfigs[i], transitionCost[i - 1])
    mdl.add(allowed_assignments(auxVars, CONFIGURATION_TRANSITION_COST))

# Constrain the volume of the orders in each truck
mdl.add(pack(load, where, volumes, nbDeliveries))
for i in range(0, maxDeliveries):
    mdl.add(load[i] <= element(truckConfigs[i], maxTruckConfigLoad))

# Compatibility between the product type of an order and the configuration of its truck
for j in range(0, nbOrders):
    configOfContainer = integer_var(ALLOWED_CONTAINER_CONFIGS[productType[j]])
    mdl.add(configOfContainer == element(truckConfigs, where[j]))

# Only one customer per delivery
for j in range(0, nbOrders):
    mdl.add(element(customerOfDelivery, where[j]) == CUSTOMER_ORDERS[j][0])

# Non-used deliveries are at the end
for j in range(1, maxDeliveries):
    mdl.add((load[j - 1] > 0) | (load[j] == 0))

# Dominance: the non used deliveries keep the last used configuration
mdl.add(load[0] > 0)
for i in range(1, maxDeliveries):
    mdl.add((load[i] > 0) | (truckConfigs[i] == truckConfigs[i - 1]))

# Dominance: regroup deliveries with same configuration
for i in range(maxDeliveries - 2, 0, -1):
    ct = true()
    for p in range(i + 1, maxDeliveries):
        ct = (truckConfigs[p] != truckConfigs[i - 1]) & ct
    mdl.add((truckConfigs[i] == truckConfigs[i - 1]) | ct)

# Objective: first criterion for minimizing the cost for configuring and loading trucks 
#            second criterion for minimizing the number of deliveries
cost = sum(transitionCost) + sum(element(truckConfigs[i], truckCost) * (load[i] != 0) for i in range(maxDeliveries))
mdl.add(minimize_static_lex([cost, nbDeliveries]))

# Search strategy: first assign order to truck
mdl.set_search_phases([search_phase(where)])

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=20)
if msol.is_solution():
    print("Solution: ")
    ovals = msol.get_objective_values()
    print("   Configuration cost: {}, number of deliveries: {}".format(ovals[0], ovals[1]))
    for i in range(maxDeliveries):
        ld = msol.get_value(load[i])
        if ld > 0:
            stdout.write("   Delivery {:2d}: config={}".format(i,msol.get_value(truckConfigs[i])))
            stdout.write(", items=")
            for j in range(nbOrders):
                if (msol.get_value(where[j]) == i):
                    stdout.write(" <{}, {}, {}>".format(j, productType[j], volumes[j]))
            stdout.write('\n')
else:
    stdout.write("Solve status: {}\n".format(msol.get_solve_status()))