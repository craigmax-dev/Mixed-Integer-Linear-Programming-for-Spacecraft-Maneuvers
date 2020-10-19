import sys
import docplex.mp

url = None
key = None

capacities = {1: 15, 2: 20}
demands = {3: 7, 4: 10, 5: 15}
costs = {(1,3): 2, (1,5):4, (2,4):5, (2,5):3}

# Python ranges will be used to iterate on source, target nodes.
source = range(1, 3) # {1, 2}
target = range(3, 6) # {3,4,5}

from docplex.mp.model import Model

tm = Model(name='transportation')

# create flow variables for each couple of nodes
# x(i,j) is the flow going out of node i to node j
x = {(i,j): tm.continuous_var(name='x_{0}_{1}'.format(i,j)) for i in source for j in target}

# each arc comes with a cost. Minimize all costed flows
tm.minimize(tm.sum(x[i,j]*costs.get((i,j), 0) for i in source for j in target))

tm.print_information()


# for each node, total outgoing flow must be smaller than available quantity
for i in source:
    tm.add_constraint(tm.sum(x[i,j] for j in target) <= capacities[i])
    
# for each target node, total ingoing flow must be greater thand demand
for j in target:
    tm.add_constraint(tm.sum(x[i,j] for i in source) >= demands[j])

tm.minimize(tm.sum(x[i,j]*costs.get((i,j), 0)))


tms = tm.solve(url=url, key=key)
assert tms
tms.display()

# create a new model to attach piecewise
pm = Model(name='pwl')
pwf1 = pm.piecewise_as_slopes([(0, 0), (0.4, 1000), (0.2, 3000)], lastslope=0.1)
# plot the function
pwf1.plot(lx=-1, rx=4000, k=1, color='b', marker='s', linewidth=2)

pwf2 = pm.piecewise(preslope=0, breaksxy=[(0, 0), (1000, 400), (3000, 800)], postslope=0.1)
# plot the function
pwf2.plot(lx=-1, rx=4000, k=1, color='r', marker='o', linewidth=2)

x = pm.continuous_var(name='x')
y = pm.continuous_var(name='y')
pm.add_constraint(y == pwf2(x));  # y is constrained to be equal to f(x)

im = Model(name='integer_programming')
b = im.binary_var(name='boolean_var')
ijk = im.integer_var(name='int_var')
im.print_information()

lm = Model(name='lp_telephone_production')
desk = lm.continuous_var(name='desk')
cell = lm.continuous_var(name='cell')
# write constraints
# constraint #1: desk production is greater than 100
lm.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
lm.add_constraint(cell >= 100)

# constraint #3: assembly time limit
ct_assembly = lm.add_constraint( 0.2 * desk + 0.4 * cell <= 401)

# constraint #4: paiting time limit
ct_painting = lm.add_constraint( 0.5 * desk + 0.4 * cell <= 492)
lm.maximize(12.4 * desk + 20.2 * cell)

ls = lm.solve(url=url, key=key)
lm.print_solution()

im = Model(name='ip_telephone_production')
desk = im.integer_var(name='desk')
cell = im.integer_var(name='cell')
# write constraints
# constraint #1: desk production is greater than 100
im.add_constraint(desk >= 100)

# constraint #2: cell production is greater than 100
im.add_constraint(cell >= 100)

# constraint #3: assembly time limit
im.add_constraint( 0.2 * desk + 0.4 * cell <= 401)

# constraint #4: paiting time limit
im.add_constraint( 0.5 * desk + 0.4 * cell <= 492)
im.maximize(12.4 * desk + 20.2 * cell)

si = im.solve(url=url, key=key)
im.print_solution()

bbm = Model(name='b&b')
x, y, z = bbm.integer_var_list(3, name=['x', 'y', 'z'])
bbm.maximize(x + y + 2*z)
bbm.add_constraint(7*x + 2*y + 3*z <= 36)
bbm.add_constraint(5*x + 4*y + 7*z <= 42)
bbm.add_constraint(2*x + 3*y + 5*z <= 28)
bbm.solve(url=url, key=key,log_output=True);

tm2 = Model('decision_phone')

# variables for total production
desk = tm2.integer_var(name='desk', lb=100)
cell = tm2.continuous_var(name='cell', lb=100)

# two variables per machine type:
desk1 = tm2.integer_var(name='desk1')
cell1 = tm2.integer_var(name='cell1')

desk2 = tm2.integer_var(name='desk2')
cell2 = tm2.integer_var(name='cell2')

# yes no variable
z = tm2.binary_var(name='z')

# total production is sum of type1 + type 2
tm2.add_constraint(desk == desk1 + desk2)
tm2.add_constraint(cell == cell1 + cell2)

# production on assembly machine of type 1 must be less than 400 if y is 1, else 0
tm2.add_constraint(0.2 * desk1 + 0.4 * cell1 <= 400 * z)
# production on assembly machine of type 2 must be less than 430 if y is 0, else 0
tm2.add_constraint(0.25 * desk2 + 0.3 * cell2 <= 430 * (1-z))

# painting machine limit is identical
# constraint #4: paiting time limit
tm2.add_constraint( 0.5 * desk + 0.4 * cell <= 490)

tm2.print_information()

tm2.maximize(12 * desk + 20 * cell)

tm2s= tm2.solve(url=url, key=key,log_output=True)
assert tm2s
tm2.print_solution()

import pandas as pd
from pandas import DataFrame

sec_data = {
    'sector': ['treasury', 'hardware', 'theater', 'telecom', 'brewery', 'highways', 'cars', 'bank', 'software',
               'electronics'],
    'return': [5, 17, 26, 12, 8, 9, 7, 6, 31, 21],
    'area': ['N-Am.', 'N-Am.', 'N-Am.', 'N-Am.', "ww", 'ww', 'ww', 'ww', 'ww', 'ww']
}

df_secs = DataFrame(sec_data, columns=['sector', 'return', 'area'])
df_secs.set_index(['sector'], inplace=True)

# store set of share names
securities = df_secs.index
df_secs

# the variance matrix
var = {
    "treasury": [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "hardware": [0, 19, -2, 4, 1, 1, 1, 0.5, 10, 5],
    "theater": [0, -2, 28, 1, 2, 1, 1, 0, -2, -1],
    "telecom": [0, 4, 1, 22, 0, 1, 2, 0, 3, 4],
    "brewery": [0, 1, 2, 0, 4, -1.5, -2, -1, 1, 1],
    "highways": [0, 1, 1, 1, -1.5, 3.5, 2, 0.5, 1, 1.5],
    "cars": [0, 1, 1, 2, -2, 2, 5, 0.5, 1, 2.5],
    "bank": [0, 0.5, 0, 0, -1, 0.5, 0.5, 1, 0.5, 0.5],
    "software": [0, 10, -2, 3, 1, 1, 1, 0.5, 25, 8],
    "electronics": [0, 5, -1, 4, 1, 1.5, 2.5, 0.5, 8, 16]
}

dfv = pd.DataFrame(var, index=securities, columns=securities)
dfv

def is_nam(s):
    return 1 if s == 'N-Am.' else 0

df_secs['is_na'] = df_secs['area'].apply(is_nam)
df_secs

from docplex.mp.advmodel import AdvModel as Model

mdl = Model(name='portfolio_miqp')

# create variables
df_secs['frac'] = mdl.continuous_var_list(securities, name='frac', ub=1)

# max fraction
frac_max = 0.3
for row in df_secs.itertuples():
    mdl.add_constraint(row.frac <= 0.3)
    
# sum of fractions equal 100%
mdl.add_constraint(mdl.sum(df_secs.frac) == 1);

# north america constraint:
#    - add a 1-0 column equal to 1 
# compute the scalar product of frac variables and the 1-0 'is_na' column and set a minimum
mdl.add_constraint(mdl.dot(df_secs.frac, df_secs.is_na) >= .4);

# ensure minimal return on investment
target_return = 9 # return data is expressed in percents
# again we use scalar product to compute compound return rate
# keep the expression to use as a kpi.
actual_return = mdl.dot(df_secs.frac, df_secs['return'])
mdl.add_kpi(actual_return, 'ROI')

# keep the constraint for later use (more on this later)
ct_return = mdl.add_constraint(actual_return >= 9);

# KPIs
fracs = df_secs.frac
variance = mdl.sum(float(dfv[sec1][sec2]) * fracs[sec1] * fracs[sec2] for sec1 in securities for sec2 in securities)
mdl.add_kpi(variance, 'Variance')

# finally the objective
mdl.minimize(variance)


assert mdl.solve(url=url, key=key), "Solve failed"
mdl.report()

all_fracs = {}
for row in df_secs.itertuples():
    pct = 100 * row.frac.solution_value
    all_fracs[row[0]] = pct
    print('-- fraction allocated in: {0:<12}: {1:.2f}%'.format(row[0], pct))

import matplotlib.pyplot as plt

def display_pie(pie_values, pie_labels, colors=None,title=''):
    plt.axis("equal")
    plt.pie(pie_values, labels=pie_labels, colors=colors, autopct="%1.1f%%")
    plt.title(title)
    plt.show()
                                                           
display_pie( list(all_fracs.values()), list(all_fracs),title='Allocated Fractions')

target_returns = range(5,21)  # from 5 to 20, included
variances = []
for target in target_returns:
    # modify the constraint's right hand side.
    ct_return.rhs = target
    cur_s = mdl.solve(url=url, key=key)
    assert cur_s  # solve is OK
    cur_variance = variance.solution_value
    print('- for a target return of: {0}%, variance={1}'.format(target, cur_variance))
    variances.append(cur_variance)

plt.plot(target_returns, variances, 'bo-')
plt.title('Variance vs. Target Return')
plt.xlabel('target return (in %)')
plt.ylabel('variance')
plt.show()

