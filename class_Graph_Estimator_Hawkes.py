##### normal libraries
import numpy as np
import statistics as stat
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time


##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions
from classes.class_estimator import *
from classes.class_graph_estimator import *

##### other files
from class_graph_hawkes import *
from class_hawkes_process import *
from class_kernel import *
import functions_change_point_analysis
import functions_fct_evol_parameters
import functions_general_for_Hawkes


class Estimator_Hawkes(Estimator):
    # DF is a dataframe from pandas. Storing information inside is quite easy, easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self):
        super().__init__(pd.DataFrame(columns=['variable', 'n', 'm',
                                  'time estimation', 'weight function',
                                  'value', 'T_max', 'true value', 'number of guesses']))



# example:
#
#  estimators = estimators.append(pd.DataFrame(
#                             {"time estimation": T[i],
#                              "variable": "alpha",
#                              "n": s,
#                              "m": t,
#                              "weight function": str(function_weight[i_weights].name),
#                              "value": ALPHA_HAT[s, t]
#                              }), sort=True
#                         )
#
# estimators = estimators.append(pd.DataFrame(
#     {"time estimation": T[i],
#      "variable": "nu",
#      "n": s,
#      "m": 0,
#      "weight function": str(function_weight[i_weights].name),
#      "value": MU_HAT[s]
#      }), sort=True
# )