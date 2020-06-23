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
from class_graph import Graph

##### other files
import functions_MLE
import class_kernel
from class_hawkes_process import *
from class_estimator_hawkes import *
import functions_general_for_Hawkes
import functions_change_point_analysis
import functions_fct_evol_parameters



# batch_estimation is one dataframe with the estimators.
class Graph_Hawkes(Graph):
    def __init__(self, estimator, fct_parameters):
        # Initialise the Graph with the estimator
        super().__init__(estimator, ['variable', 'm', 'n'])

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
        # get the max value which is M-1
        return cls(estimator, parameters)

    def get_range(self, key, mean):
        variable = key[0]
        if variable == "nu":
            return (0, 2 * mean)
        else:
            return (0.5 * mean, 1.5 * mean)

    # TODO: make more general -- don't assume that the name will always be the first
    def get_param_info(self, key, mean):
        range = self.get_range(key, mean)
        param_dict = {'bins': 30,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': range,
                      'cumulative': True
                      }
        return param_dict

    def get_fig_dict_hist(self, separators, key):
        title = self.generate_title(separators, key)
        fig_dict = {'title': "Histogram" + title,
                    'xlabel': 'value',
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict

    def get_fig_dict_plot(self, separators, key):
        title = self.generate_title(separators, key,
                                    "Only 10-90% of the interval is shown (boundary effect), starting from 0 until {}.",
                                    [self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation" + title,
                    'xlabel': 'Time',
                    'ylabel': 'Value'}
        return fig_dict



    def get_extremes(self, data):
        estimation = data['time estimation'].unique()
        values = data.groupby(['time estimation'])['value']
        return (values.min(), values.max(), estimation)
    #### create another init that takes the same parameter, with the diff that it takes the path.
    # another constructor :

    def get_true_values(self, data):
        return data.groupby(['time estimation'])['true value'].mean().values

    def get_plot_data(self, data):
        return data.groupby(['time estimation'])['value'].mean().values

    # work-in-progress
    def MSE_convergence_estimators_limit_time(self, mini_T, times):
        # BIANCA-HERE you see, using dict is a good trick for upgrading to multi dimensional case.
        #  I did that in the function written in "functions_change_point_analysis", which could be put in class graph.
        #  The problem is that my functions in plot_functions take arrays, not dictionnaries. How to convert efficiently ?

        estim_alpha = Estimator_Hawkes()
        estim_alpha.append(self.estimator.DF.loc[self.estimator.DF['variable'] == "alpha"].copy())
        estim_beta = Estimator_Hawkes()
        estim_beta.append(self.estimator.DF.loc[self.estimator.DF['variable'] == "beta"].copy())
        estim_nu = Estimator_Hawkes()
        estim_nu.append(self.estimator.DF.loc[self.estimator.DF['variable'] == "nu"].copy())

        # test si MEAN est ok
        if estim_alpha.DF["true value"].nunique() != 1 or \
                estim_beta.DF["true value"].nunique() != 1 or \
                estim_nu.DF["true value"].nunique() != 1:
            raise (
                "Error because you are estimating different parameters, but still compounding the MSE error together.")
        estim_alpha.function_upon_separeted_data("value", recurrent_functions.compute_MSE, "compute_MSE",
                                                 true_parameter=estim_alpha.DF["true value"].mean())
        estim_beta.function_upon_separeted_data("value", recurrent_functions.compute_MSE, "compute_MSE",
                                                true_parameter=estim_beta.DF["true value"].mean())
        estim_nu.function_upon_separeted_data("value", recurrent_functions.compute_MSE, "compute_MSE",
                                              true_parameter=estim_nu.DF["true value"].mean())

        DF_MSE = estim_nu.DF.groupby(['T_max'])["compute_MSE"].sum()

        MSE_reals = np.zeros(len(times))
        TIMES_plot = [times[i] // mini_T * 50 for i in range(len(times))]
        i = 0
        for time in times:
            MSE_reals[i] = DF_MSE[time] / self.nb_of_guesses
            i += 1

        plot_functions.plot_graph(TIMES_plot, MSE_reals, title=[
            "Convergence in compute_MSE of the estimators, batches of {} realisations.".format(self.nb_of_guesses)],
                                  labels=["Nb of Events", "compute_MSE of the Estimator"],
                                  parameters=[self.ALPHA[0][0](0, 1), self.BETA[0][0](0, 1), self.NU[0](0, 1)],
                                  name_parameters=["ALPHA", "BETA", "NU"],
                                  name_save_file="MSE_convergence")

        # TODO I NEED THE HISTOGRAM OF LAST VALUE IN compute_MSE...
        return
