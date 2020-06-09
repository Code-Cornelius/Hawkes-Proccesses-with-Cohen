# https://github.com/amanahuja/change-detection-tutorial/blob/master/ipynb/section_tmp_all.ipynb
# https://github.com/deepcharles/ruptures


import numpy as np
import statistics as stat
import pandas as pd

import functions_MLE
import scipy.stats

from generic_functions import *
import plot_functions
from useful_functions import *

from operator import itemgetter  # at some point I need to get the list of ranks of a list.

import class_kernel

from class_hawkes_process import *
from class_estimator import *
from class_graph import *

import ruptures as rpt
import functions_general_for_Hawkes



def change_point_plot(path, width, min_size, n_bkps = 1, model = "l2",  column_for_multi_plot_name = None):
    # number of breakpoints doesn't support a different value of bkps for each variable.
    # path should be with \\
    # path is where the file is located
    # column_for_multi_plot_name a string
    estimator = Estimator(pd.read_csv(path))
    # get the max value which is M-1
    M = estimator.DF["m"].max() + 1

    if column_for_multi_plot_name is not None:
        list_of_unique_elements = estimator.DF[column_for_multi_plot_name].unique()
        different_estimator_categories = []
        for name_unique in list_of_unique_elements:
            different_estimator_categories.append(estimator.DF[estimator.DF[column_for_multi_plot_name] == name_unique].copy())
    else :  # otherwise I just need to look at the whole DF
        list_of_unique_elements = estimator.DF[column_for_multi_plot_name].unique()
        different_estimator_categories = estimator.DF


    dict_of_estimators = {}
    for counter, filtered_DF in enumerate(different_estimator_categories):
        for i in range(M): #m
            for j in range(M): #n
                key = ('alpha',i,j, list_of_unique_elements[counter]) #( alpha, m, n, kernel)
                value = (different_estimator_categories[counter])[  # list of the conditions for that value; no condition on the last differenting because they already have been filtered.
                    ((different_estimator_categories[counter])['variable'] == "alpha") &
                    ((different_estimator_categories[counter])['m'] == i) &
                    ((different_estimator_categories[counter])['n'] == j)
                ].copy()

                dictionary = {key: value}
                dict_of_estimators.update(dictionary)

                key = ('beta', i, j, list_of_unique_elements[counter])  # ( alpha, m, n, kernel)
                value = (different_estimator_categories[counter])[  # list of the conditions for that value; no condition on the last differenting because they already have been filtered.
                    ((different_estimator_categories[counter])['variable'] == "beta") &
                    ((different_estimator_categories[counter])['m'] == i) &
                    ((different_estimator_categories[counter])['n'] == j)
                ].copy()

                dictionary = {key: value}
                dict_of_estimators.update(dictionary)
            #I put this after the two loops, but still in the i's;
            key = ( 'mu',i,0, list_of_unique_elements[counter] )
            value = (different_estimator_categories[counter])[
                # list of the conditions for that value; no condition on the last differenting because they already have been filtered.
                ((different_estimator_categories[counter])['variable'] == "nu") &
                ((different_estimator_categories[counter])['m'] == i)
                ].copy()
            dictionary = {key: value}
            dict_of_estimators.update(dictionary)
    # at this stage, I have a M*M*3 of DF. Now, on each of these elements, I apply the transformation, which will allow me to have my time series.

    # The way I am doing it, doesn't ensure that the plot will always be in the same order unfort.
    for k in dict_of_estimators.keys(): # iterate through dictionary
        dict_of_estimators[k] = dict_of_estimators[k].groupby(['time estimation'])['value'].mean().values  # I update every entry by its time series.
    dict_of_times_series = {}
    for k1,k2,k3,k4 in dict_of_estimators.keys(): # iterate through dictionary
        if (k1,k2,k3) not in dict_of_times_series: # not yet crossed those values
            dict_of_times_series[(k1, k2, k3)] = dict_of_estimators[(k1, k2, k3,k4)]
        else : # the condition already seen, so I aggregate to what was already done.
            dict_of_times_series[(k1, k2, k3)] = np.vstack( (dict_of_times_series[(k1, k2, k3)],
                                                            dict_of_estimators[(k1, k2, k3, k4)]
                                                            ) )
    for k in dict_of_times_series.keys():  # iterate through dictionary
        dict_of_times_series[ k ] = np.transpose(dict_of_times_series[k])



    model = model
    ############################################## dynamic programming   http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/dynp.html
    for k in dict_of_times_series.keys():
        algo = rpt.Dynp(model=model, min_size=min_size, jump= 1 ).fit(dict_of_times_series[k])
        my_bkps1 = algo.predict(n_bkps=n_bkps)
        rpt.show.display(dict_of_times_series[k], my_bkps1, figsize=(10, 6))
        algo = rpt.Window(width=width, model=model).fit(dict_of_times_series[k])
        my_bkps1 = algo.predict(n_bkps=1)
        rpt.show.display(dict_of_times_series[k], my_bkps1, figsize=(10, 6), title = "g")
    plt.show()




