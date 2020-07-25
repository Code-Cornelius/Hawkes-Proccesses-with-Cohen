##### normal libraries
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import numpy as np  # maths library and arrays
import statistics as stat
import pandas as pd  # dataframes
import seaborn as sns  # envrionement for plots
from matplotlib import pyplot as plt  # ploting
import scipy.stats  # functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time  # allows to time event
import warnings
import math  # quick math functions
import cmath  # complex functions
from inspect import signature  # used in the method eval of the class

##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions
from classes.class_estimator import *
from classes.class_graph_estimator import *

np.random.seed(124)


##### other files


# -------------------------------------------------------------------------------------------------------
# list of the possible kernels:
# fct_top_hat
# fct_plain
# fct_truncnorm
# fct_biweight
#
#
#### the functions are correct, they scale and shift the way it is supposed.
# However they are written in the following way : f_t(t_i) = K( t_i - t )

# example of kernels:
# list_of_kernels = [Kernel(fct_top_hat, name="wide top hat", a=-450, b=450),
#                    Kernel(fct_top_hat, name="normal top hat", a=-200, b=200),
#                    Kernel(fct_truncnorm, name="wide truncnorm", a=-500, b=500, sigma=350),
#                    Kernel(fct_truncnorm, name="normal truncnorm", a=-350, b=350, sigma=250)]
# -------------------------------------------------------------------------------------------------------
# the functions only work for positive time. If one input negative times, it messes up the orientation.
class Kernel:
    # kernel is a class of objects, where using eval evaluates the function given as parameter
    # the evaluation gives back a list of np.array
    # the function should hand in the list of np.arrays non scaled.
    # list of arrays allow me to perform computations on each faster.

    # I also chose to not give T_t to the kernel in order to make it a function of the T_t (and of where one evaluates the kernel),
    # this allows me to fix the parameters kwargs of the kernel upfront.
    # same for eval_point.
    # Those parameters are the given kwargs.

    # the array gathers the weights held by data wrt when we do estimate.

    # the name is for identification in plots
    def __init__(self, fct_kernel, name=' no name ', **kwargs):
        self.fct_kernel = fct_kernel
        self.name = name
        self.__dict__.update(kwargs)

    def __repr__(self):
        return repr(self.fct_kernel)

    def eval(self, T_t, eval_point, T_max):
        length_elements_T_t = [len(T_t[i]) for i in range(len(T_t))]
        # ans is the kernel evaluated on the jumps
        ans = self.fct_kernel(T_t=T_t, eval_point=eval_point, length_elements_T_t=length_elements_T_t,
                              **{k: self.__dict__[k] for k in self.__dict__ if
                                 k in signature(self.fct_kernel).parameters})
        # ans is a list of np arrays. It is normalized such that it is a kernel.
        # then I want to scale every vector.
        # The total integral should be T_max, so I multiply by T_max

        # If it isn't fct plain, then I have to scale.
        if self.fct_kernel.__name__ != 'fct_plain':

            # I want to rescale the results for the kernels that are not covering seen part. For that reason,
            # I compute the percent of the kernel that is not used, and I multiply the result by that value.
            lower_a = self.__dict__["a"]
            upper_b = self.__dict__["b"]
            interval_a_b = upper_b - lower_a
            not_enough_left = eval_point + lower_a
            too_much_right = eval_point + upper_b - T_max

            coef_1 = coef_2 = 1
            if not_enough_left < 0:
                # the number is negative so I add it.
                coef_1 += not_enough_left / interval_a_b

            if too_much_right > 0:
                coef_2 -= too_much_right / interval_a_b
            #coef1 and 2 are the rescaling factors corresponding to the percent of used data over the width of the kernel.

            if self.fct_kernel.__name__ == 'fct_truncnorm_test':
                coef_1 = coef_2 = 1

            for i in range(len(length_elements_T_t)):
                # / np.sum(ans[i])
                ans[i] = ans[i]  * T_max / coef_1 / coef_2  # *= do not work correctly since the vectors are not the same type (int/float).
                # I also divide by the sum, the vector is normalized, however, possibly we're on the edge and we need to take that into account.
        # print("inside kernel debug, that's my integral : ", np.sum(ans[i][:-1] ) * T_max / (len(ans[i])-1))
        return ans


def fct_top_hat(T_t, length_elements_T_t, eval_point, a=-200, b=200):
    output = []
    for i in range(len(length_elements_T_t)):
        vector = np.array(T_t[i])
        # -1 if x < 0, 0 if x==0, 1 if x > 0.
        output.append(1 / (2 * (b - a)) * \
                      (np.sign(vector - eval_point - a) +
                       np.sign(b - vector + eval_point))
                      )
    return output


def fct_plain(T_t, length_elements_T_t, eval_point):
    # no scaling parameter, would be full to use scaling on plain.
    return [
        np.full(length_elements_T_t[i], 1)
        for i in range(len(length_elements_T_t))
    ]


def fct_truncnorm(T_t, length_elements_T_t, eval_point, a=-300, b=300, sigma=200):
    output = []
    for i in range(len(length_elements_T_t)):
        output.append(scipy.stats.truncnorm.pdf(np.array(T_t[i]), (a) / sigma, (b) / sigma,
                                                loc=eval_point, scale=sigma))
    return output




def fct_truncnorm_test(T_t, length_elements_T_t, eval_point, a=-300, b=300, sigma=200):
    output = []
    for i in range(len(length_elements_T_t)):
        output.append( 2 * scipy.stats.truncnorm.pdf(np.array(T_t[i]), (a) / sigma, (b) / sigma,
                                                loc=eval_point, scale=sigma))
    output[i][
        T_t[i] < eval_point
        ] = 0
    return output

#  if important, I can generalize biweight with function beta.
#  Thus creating like 4 kernels with one function ( BETA(1), BETA(2)...)
def fct_biweight(T_t, length_elements_T_t, eval_point, a=-300, b=300, scaling_vect=None):
    output = []
    for i in range(len(length_elements_T_t)):
        xx = (np.array(T_t[i]) - (a + b) / 2 - eval_point) * 2 / (b - a)
        # the correct order is eval_point - T_t,
        # bc we evaluate at eval_point but translated by T_t,
        # if kernel not symmetric a != b, then we also need to translate by the mid of them.
        xx[(xx < -1) | (xx > 1)] = 1
        output.append(15 / 16 * np.power(1 - xx * xx, 2) * 2 / (b - a))  # kernel * scaling ; delta in my formulas
    return output



def fct_epa(T_t, length_elements_T_t, eval_point, a=-300, b=300):
    output = []
    for i in range(len(length_elements_T_t)):
        xx = (np.array(T_t[i]) - (a + b) / 2 - eval_point) * 2 / (b - a)
        # the correct order is eval_point - T_t,
        # bc we evaluate at eval_point but translated by T_t,
        # if kernel not symmetric a != b, then we also need to translate by the mid of them.
        xx[(xx < -1) | (xx > 1)] = 1
        output.append(3 / 4 * (1 - xx * xx) * 2 / (b - a))  # kernel * scaling ; delta in my formulas
    return output













# thre versions :

# where on coupe en 3
# if et vectorize ?
# autre ?
def test_geom_kern(T_t, G = 10, min = None, max=None):
    if min is None:
        min = np.quantile(T_t, 0.02)
    print(min)
    if max is None:
        max = np.quantile(T_t, 0.75)
    print(max)
    output = []
    xx = T_t[i] - G
    #xx[ (xx < -math.pi) | (xx > math.pi) ] = math.pi

    ans = 0
    scaling1 =  math.pi / (G - min)
    scaling2 =  math.pi / (max - G)
    # I fix outside of my interest array to be the final value, which is math.pi.
    # I also need the scaling by +50 given by math.pi
    my_xx2 = np.where((xx*scaling1 > -math.pi) & (xx*scaling1 < 0),
                           xx*scaling1, math.pi) # left
    my_xx3 = np.where((xx*scaling2 > 0) & (xx*scaling2 < math.pi ),
                           xx*scaling2, math.pi) # right
    ans += - 50 * np.cos( my_xx2  )
    ans += - 50 * np.cos( my_xx3  )

    ans += 0.01 # avoid infinite width kernel
    output.append( ans )
    return output


def test_normal_kernel(T_t, G = 10., gamma = 0.5):
    output = []
    xx = T_t[0]
    print(xx)
    ans = np.power(xx / G,-gamma)
    print(ans)
    output.append( ans )
    return output




























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



# ############ test
# T_t = [np.linspace(0,2000,10000)]
# aplot = APlot(how = (1,1))
# aplot.set_dict_fig(0, {'title':"", 'xlabel':"", 'ylabel':""})
#
#
# color = plt.cm.Dark2.colors
# for fct,c in zip([fct_top_hat, fct_truncnorm, fct_truncnorm_test],color):
#     my_kernel = Kernel(fct, a=-500, b=500)
#     length_elements_T_t = [10000]
#     eval_point = [0,1000,2000]
#     for i in eval_point:
#         res = my_kernel.eval( T_t, i, 2000)
#         aplot.uni_plot(nb_ax=0, xx=T_t[0], yy=res[0], dict_plot_param={"color":c, "label":fct.__name__,"markersize" : 0, "linewidth":2})
# aplot.show_legend()
#
#
#
# plt.show()
