import sys
import docplex.mp
import cplex

# Teams in 1st division
team_div1 = ["Baltimore Ravens","Cincinnati Bengals", "Cleveland Browns","Pittsburgh Steelers","Houston Texans",
                "Indianapolis Colts","Jacksonville Jaguars","Tennessee Titans","Buffalo Bills","Miami Dolphins",
                "New England Patriots","New York Jets","Denver Broncos","Kansas City Chiefs","Oakland Raiders",
                "San Diego Chargers"]

# Teams in 2nd division
team_div2 = ["Chicago Bears","Detroit Lions","Green Bay Packers","Minnesota Vikings","Atlanta Falcons",
                "Carolina Panthers","New Orleans Saints","Tampa Bay Buccaneers","Dallas Cowboys","New York Giants",
                "Philadelphia Eagles","Washington Redskins","Arizona Cardinals","San Francisco 49ers",
                "Seattle Seahawks","St. Louis Rams"]

#number_of_matches_to_play = 1  # Number of match to play between two teams on the league
# Schedule parameters
nb_teams_in_division = 8
max_teams_in_division = 16
number_of_matches_inside_division = 1
number_of_matches_outside_division = 1

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

import pandas as pd

team1 = pd.DataFrame(team_div1)
team2 = pd.DataFrame(team_div2)
team1.columns = ["AFC"]
team2.columns = ["NFC"]

teams = pd.concat([team1,team2], axis=1)

from IPython.display import display

display(teams)

import numpy as np

nb_teams = 2 * nb_teams_in_division
teams = range(nb_teams)

# Calculate the number of weeks necessary
nb_weeks = (nb_teams_in_division - 1) * number_of_matches_inside_division \
        + nb_teams_in_division * number_of_matches_outside_division


# Weeks to schedule
weeks = range(nb_weeks)

# Season is split into two halves
first_half_weeks = range(int(np.floor(nb_weeks / 2)))
nb_first_half_games = int(np.floor(nb_weeks / 3))

from collections import namedtuple

match = namedtuple("match",["team1","team2","is_divisional"])

matches = {match(t1,t2, 1 if ( t2 <= nb_teams_in_division or t1 > nb_teams_in_division) else 0)
           for t1 in teams for t2 in teams if t1 < t2}

nb_play = { m :  number_of_matches_inside_division if m.is_divisional==1
                                                   else number_of_matches_outside_division
                   for m in matches}

from docplex.mp.environment import Environment
env = Environment()
env.print_information()

from docplex.mp.model import Model

mdl = Model("sports")

plays = mdl.binary_var_matrix(matches, weeks, lambda ij: "x_%s_%d" %(str(ij[0]), ij[1]))

mdl.add_constraints( mdl.sum(plays[m,w]  for w in weeks) == nb_play[m]
                   for m in matches)
mdl.print_information()

mdl.add_constraints( mdl.sum(plays[m,w] for m in matches if (m.team1 == t or m.team2 == t) )  == 1
                   for w in weeks for t in teams)
mdl.print_information()

mdl.add_constraints( plays[m,w] + plays[m,w+1] <= 1
                   for w in weeks
                   for m in matches
                   if w < nb_weeks-1)
mdl.print_information()

mdl.add_constraints( mdl.sum(plays[m,w]  for w in first_half_weeks for  m in matches
                            if (((m.team1 == t or m.team2 == t) and m.is_divisional == 1 )))
                    >= nb_first_half_games
                   for t in teams)
mdl.print_information()

gain = { w : w*w for w in weeks}

# If an intradivisional pair plays in week w, Gain[w] is added to the objective.
mdl.maximize( mdl.sum (m.is_divisional * gain[w] * plays[m,w] for m in matches for w in weeks) )

mdl.print_information()

assert mdl.solve(), "!!! Solve of the model fails"
mdl.report()

