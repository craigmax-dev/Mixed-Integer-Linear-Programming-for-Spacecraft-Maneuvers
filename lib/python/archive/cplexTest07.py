import sys
import docplex.mp
from docplex.mp.model import Model # Import model class

url = None
key = None

# Create model instance
m = Model(name='Satellite Trajectory Optimization')

# by default, all variables in Docplex have a lower bound of 0 and infinite upper bound
desk = m.continuous_var(name='desk')
cell = m.continuous_var(name='cell')