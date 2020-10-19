import sys
import pandas as pd
import docplex.mp

CSS = """
body {
    margin: 0;
    font-family: Helvetica;
}
table.dataframe {
    border-collapse: collapse;
    border: none;
}
table.dataframe tr {
    border: none;
}
table.dataframe td, table.dataframe th {
    margin: 0;
    border: 1px solid white;
    padding-left: 0.25em;
    padding-right: 0.25em;
}
table.dataframe th:not(:empty) {
    background-color: #fec;
    text-align: left;
    font-weight: normal;
}
table.dataframe tr:nth-child(2) th:empty {
    border-left: none;
    border-right: 1px dashed #888;
}
table.dataframe td {
    border: 2px solid #ccf;
    background-color: #f4f4ff;
}
    table.dataframe thead th:first-child {
        display: none;
    }
    table.dataframe tbody th {
        display: none;
    }
"""
from IPython.core.display import HTML
HTML('<style>{}</style>'.format(CSS))

from IPython.display import display
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen
# This notebook requires pandas to work
import pandas as pd
from pandas import DataFrame
import xlrd

# Use pandas to read the file, one tab for each table.
data_url = "https://github.com/IBMDecisionOptimization/docplex-examples/blob/master/examples/mp/jupyter/nurses_data.xls?raw=true"
nurse_xls_file = pd.ExcelFile(urlopen(data_url))

df_skills = nurse_xls_file.parse('Skills')
df_depts  = nurse_xls_file.parse('Departments')
df_shifts = nurse_xls_file.parse('Shifts')
# Rename df_shifts index
df_shifts.index.name = 'shiftId'

# Index is column 0: name
df_nurses = nurse_xls_file.parse('Nurses', header=0, index_col=0)
df_nurse_skilles = nurse_xls_file.parse('NurseSkills')
df_vacations = nurse_xls_file.parse('NurseVacations')
df_associations = nurse_xls_file.parse('NurseAssociations')
df_incompatibilities = nurse_xls_file.parse('NurseIncompatibilities')

# Display the nurses dataframe
print("#nurses = {}".format(len(df_nurses)))
print("#shifts = {}".format(len(df_shifts)))
print("#vacations = {}".format(len(df_vacations)))

# maximum work time (in hours)
max_work_time = 40

# maximum number of shifts worked in a week.
max_nb_shifts = 5

df_shifts