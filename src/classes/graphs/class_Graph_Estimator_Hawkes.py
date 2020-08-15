##### normal libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
##### my libraries

# errors:
import plot_functions
from errors import Error_forbidden
from errors.Warning_deprecated import deprecated_function

##### other files
from classes.class_Estimator_Hawkes import *
from classes.class_kernel import *


# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    evolution_name = 'time estimation'

    def __init__(self, estimator, fct_parameters):
        deprecated_function(reason="Graph_Estimator_Hawkes"
                                   "")
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator
        Graph_Estimator.__init__(self, estimator, ['parameter', 'm', 'n'])

        # parameters is a list of lists of lists of functions
        self.ALPHA = fct_parameters[1]
        self.BETA = fct_parameters[2]  # makes the file more readable.
        self.NU = fct_parameters[0]
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(self.BETA))
        self.T_max = estimator.DF["T_max"].max()
        self.M = np.shape(self.ALPHA)[1]
        self.nb_of_guesses = estimator.DF['number of guesses'].max()

    @classmethod
    def from_path(cls, path, parameters):
        # path has to be raw. with \\
        estimator = Estimator_Hawkes()
        estimator.append(pd.read_csv(path))
        return cls(estimator, parameters)