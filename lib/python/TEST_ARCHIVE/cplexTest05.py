import sys
import docplex.mp

url = None
key = None

# first import the Model class from docplex.mp
from docplex.mp.model import Model

# create one model instance, with a name
m = Model(name='telephone_production')

# by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
desk = m.continuous_var(name='desk')
cell = m.continuous_var(name='cell')

# write constraints
# constraint #1: desk production is greater than 100
m.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
m.add_constraint(cell >= 100)

# constraint #3: assembly time limit
ct_assembly = m.add_constraint( 0.2 * desk + 0.4 * cell <= 400)

# constraint #4: paiting time limit
ct_painting = m.add_constraint( 0.5 * desk + 0.4 * cell <= 490)

m.maximize(12 * desk + 20 * cell)

m.print_information()

s = m.solve(url=url, key=key)
m.print_solution()

# create a new model, copy of m
im = m.copy()
# get the 'desk' variable of the new model from its name
idesk = im.get_var_by_name('desk')
# add a new (infeasible) constraint
im.add_constraint(idesk >= 1100);
# solve the new proble, we expect a result of None as the model is now infeasible
ims = im.solve(url=url, key=key)
if ims is None:
    print('- model is infeasible')

overtime = m.continuous_var(name='overtime', ub=40)
ct_assembly.rhs = 400 + overtime
m.maximize(12*desk + 20 * cell - 2 * overtime)
s2 = m.solve(url=url, key=key)
m.print_solution()

print('* desk variable has reduced cost: {0}'.format(desk.reduced_cost))
print('* cell variable has reduced cost: {0}'.format(cell.reduced_cost))

# revert soft constraints
ct_assembly.rhs = 440
s3 = m.solve(url=url, key=key)

# now get slack value for assembly constraint: expected value is 40
print('* slack value for assembly time constraint is: {0}'.format(ct_assembly.slack_value))
# get slack value for painting time constraint, expected value is 0.
print('* slack value for painting time constraint is: {0}'.format(ct_painting.slack_value))

m.parameters.lpmethod = 4
m.solve(url=url, key=key, log_output=True)