# normal libraries
import numpy as np  #maths library and arrays
import statistics as stat
import pandas as pd  #dataframes
import seaborn as sns  #envrionement for plots
from matplotlib import pyplot as plt  #ploting 
import scipy.stats  #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time  #allows to time event
import warnings
import math  #quick math functions
import cmath  #complex functions

from scipy.stats.mstats import gmean


# my libraries
import classical_functions
import decorators_functions
import financial_functions
import functions_networkx
import plot_functions
import recurrent_functions
import errors.error_convergence
import classes.class_estimator
import classes.class_graph_estimator

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_geom_kern(times, G=10, L=None, R=None, h=100, l=0.01):
    if L is None:
        L = np.quantile(times, 0.02)
    print("Left boundary : ", L)
    if R is None:
        R = np.quantile(times, 0.75)
    print("Right boundary : ", R)
    output = []
    xx = times - G
    #xx[ (xx < -math.pi) | (xx > math.pi) ] = math.pi

    ans = 0
    scaling1 =  math.pi / (G - L)
    scaling2 =  math.pi / (R - G)
    # I fix the part outside of my interest, to be the final value, h. This part corresponds to math.pi.
    # I also need the scaling by +50 given by math.pi

    #xx2 and xx3 are the cosinus, but they are different cosinus.
    # So I fix them where I don't want them to move at 0 and then I can add the two functions.
    my_xx2 = np.where((xx*scaling1 > -math.pi) & (xx*scaling1 < 0),
                           xx*scaling1, math.pi) # left
    my_xx3 = np.where((xx*scaling2 > 0) & (xx*scaling2 < math.pi ),
                           xx*scaling2, math.pi) # right
    ans += - (h-l)/2 * np.cos( my_xx2  )
    ans += - (h-l)/2 * np.cos( my_xx3  )

    ans += l # avoid infinite width kernel
    output.append( ans )
    return output


def test_normal_kernel(times, G=10., gamma=0.5):
    output = []
    xx = times
    print(xx)
    ans = np.power(xx / G,-gamma)
    print(ans)
    output.append( ans )
    return output



def rescaling(times, first_estimate):
    # on each row should be one estimate, on each column one time.
    # todo norm of first estimate
    ans = np.zeros(len(times))
    #ans is my vector of normed estimates. Each value is for one time.

    for counter, time in enumerate(times):
        intermediate_vector = first_estimate[counter, :]
        ans[counter] = np.linalg.norm(intermediate_vector, 2)

    # I compute the geometric mean from our estimator.
    G = gmean(ans)
    print("G : ", G)
    scaling_factors = test_geom_kern(ans, G=G)
    return scaling_factors

my_estimator = [[1.1,1.2,1.3],[1,1.1,1.1], [1.2,1.2,1.2],[2,2,2]]
my_estimator = np.array(my_estimator)
ans = rescaling(np.linspace(0,1000,4), my_estimator)
print(ans)

############ test adaptive window
# T_t = [np.linspace(0.1,100,10000)]
# G = 10.
# #T_t = [np.random.randint(0,6*G, 20)]
# eval_point = [0]
# for i in eval_point:
#     min = np.quantile(T_t, 0.02)
#     max = np.quantile(T_t, 0.75)
#     res = test_geom_kern(T_t, G, min = min, max = max)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = T_t[0], yy = res[0])
#     aplot.plot_vertical_line(G, np.linspace(-5,105, 1000), nb_ax=0, dict_plot_param={'color':'k', 'linestyle':'--', 'markersize':0, 'linewidth':2, 'label':'geom. mean'})
#     aplot.plot_vertical_line(min, np.linspace(-5, 105, 1000), nb_ax=0,
#                              dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2, 'label':'lower bound'})
#     aplot.plot_vertical_line(max, np.linspace(-5, 105, 1000), nb_ax=0,
#                              dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2, 'label':'upper bound'})
#     aplot.set_dict_fig(0, {'title':'Adaptive scaling for Adaptive Window Width','xlabel':'Value', 'ylabel':'Scaling'})
#     aplot.show_legend()
#
# eval_point = [0]
# for i in eval_point:
#     res = test_normal_kernel(T_t, G, gamma = 0.5)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = T_t[0], yy = res[0])
#     aplot.plot_vertical_line(G, np.linspace(-1,10, 1000), nb_ax=0, dict_plot_param={'color':'k', 'linestyle':'--', 'markersize':0, 'linewidth':2, 'label':'geom. mean'})
#     aplot.set_dict_fig(0, {'title':'Adaptive scaling for Adaptive Window Width','xlabel':'Value', 'ylabel':'Scaling'})
#     aplot.show_legend()