# https://github.com/amanahuja/change-detection-tutorial/blob/master/ipynb/section_tmp_all.ipynb
# https://github.com/deepcharles/ruptures


##### normal libraries
import numpy as np
import statistics as stat
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time
import ruptures as rpt


##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions

##### other files
import functions_MLE
import class_kernel
from class_hawkes_process import *
from class_estimator_hawkes import *
from class_graph_hawkes import *
import functions_general_for_Hawkes
import functions_fct_evol_parameters



def change_point_plot(path, width, min_size, n_bkps=1, model="l2", column_for_multi_plot_name=None):
    # number of breakpoints doesn't support a different value of bkps for each variable.
    # path should be with \\
    # path is where the file is located
    # column_for_multi_plot_name a string
    estimator = Estimator_Hawkes()
    estimator.append(pd.read_csv(path))
    # get the max value which is M-1
    M = estimator.DF["m"].max() + 1



    #################################### my code version
    separators = ['variable', 'm', 'n']

    dict_serie = {}
    global_dict = estimator.DF.groupby(separators)
    for k1, k2, k3 in global_dict.groups.keys():
        if column_for_multi_plot_name is not None:
            super_dict = global_dict.get_group((k1, k2, k3)).groupby([column_for_multi_plot_name])
            for k4 in super_dict.groups.keys():
                # discrimination of whether the serie already exists.
                if (k1, k2, k3) not in dict_serie:  # not yet crossed those values
                    dict_serie[(k1, k2, k3)] = super_dict.get_group(k4).groupby(['time estimation'])[
                        'value'].mean().values.reshape((1, -1))
                else:  # the condition already seen, so I aggregate to what was already done.
                    dict_serie[(k1, k2, k3)] = np.vstack((dict_serie[(k1, k2, k3)],
                                                          super_dict.get_group(k4).groupby(['time estimation'])[
                                                              'value'].mean()
                                                          ))
        else:
            dict_serie[(k1, k2, k3)] = global_dict.get_group((k1, k2, k3)).groupby(['time estimation'])[
                'value'].mean().values.reshape((1, -1))

    for k in dict_serie.keys():  # iterate through dictionary
        dict_serie[k] = np.transpose(dict_serie[k])

    model = model
    ############################################## dynamic programming   http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/dynp.html
    for k in dict_serie.keys():
        algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(dict_serie[k])
        my_bkps1 = algo.predict(n_bkps=n_bkps)
        rpt.show.display(dict_serie[k], my_bkps1, figsize=(10, 6))
        algo = rpt.Window(width=width, model=model).fit(dict_serie[k])
        my_bkps1 = algo.predict(n_bkps=1)
        rpt.show.display(dict_serie[k], my_bkps1, figsize=(10, 6))
    plt.show()
