##### normal libraries
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import numpy as np #maths library and arrays
import statistics as stat
import pandas as pd #dataframes
import seaborn as sns #envrionement for plots
from matplotlib import pyplot as plt #ploting
import scipy.stats #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time #allows to time event
import warnings
import math #quick math functions
import cmath  #complex functions
from inspect import signature #used in the method eval of the class

##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions
from classes.class_estimator import *
from classes.class_graph_estimator import *
np.random.seed(124)

##### other files





#-------------------------------------------------------------------------------------------------------
# list of the possible kernels:
            # fct_top_hat
            # fct_plain
            # fct_truncnorm
            # fct_biweight
            #
            #


#example of kernels:
# list_of_kernels = [Kernel(fct_top_hat, name="wide top hat", a=-450, b=450),
#                    Kernel(fct_top_hat, name="normal top hat", a=-200, b=200),
#                    Kernel(fct_truncnorm, name="wide truncnorm", a=-500, b=500, sigma=350),
#                    Kernel(fct_truncnorm, name="normal truncnorm", a=-350, b=350, sigma=250)]
#-------------------------------------------------------------------------------------------------------
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
        # optimize cool to optimize using numpy and vectorize.
        length_elements_T_t = [len(T_t[i]) for i in range(len(T_t))]
        # ans is the kernel evaluated on the jumps
        ans = self.fct_kernel(T_t=T_t, eval_point=eval_point, length_elements_T_t=length_elements_T_t,
                              **{k: self.__dict__[k] for k in self.__dict__ if
                                 k in signature(self.fct_kernel).parameters})
        # ans is a list of np arrays.
        # then I want to scale every vector.
        # The total integral should be T_max, so I multiply by T_max

        # If it isn't fct plain, then I have to scale.
        if self.fct_kernel.__name__ != 'fct_plain':
            for i in range(len(length_elements_T_t)):
                ans[i] = ans[i] * T_max  # *= do not work correctly since the vectors are not the same type (int/float).
        return ans











def fct_top_hat(T_t, length_elements_T_t, eval_point, a=-200, b=200, scaling_vect=None):
    if scaling_vect is None:
        output = []
        for i in range(len(length_elements_T_t)):
            vector = T_t[i]
            # TODO vector is np array?
            # sign is -1 negative, 1 positive
            output.append(1 / (2 * (b - a)) * \
                          (np.sign(vector - eval_point - a) +
                           np.sign(b - vector + eval_point))
                          )
    else:
        # optimize cool to optimize using numpy and vectorize.
        output = [[] for _ in range(len(length_elements_T_t))]
        for i in range(len(length_elements_T_t)):
            vector = T_t[i]
            for j in range(length_elements_T_t[i]):
                a_scaled = a / scaling_vect[i][j]
                b_scaled = b / scaling_vect[i][j]
                output[i].append(1 / (2 * (b_scaled - a_scaled)) * \
                              (np.sign(vector[j] - eval_point - a_scaled) +
                               np.sign(b_scaled - vector[j] + eval_point))
                              )
            output[i] = np.array(output[i])
    return output


def fct_plain(T_t, length_elements_T_t, eval_point):
    # no scaling parameter, would be full to use scaling on plain.
    return [
        np.full(length_elements_T_t[i], 1)
        for i in range(len(length_elements_T_t))
    ]


def fct_truncnorm(T_t, length_elements_T_t, eval_point, a=-300, b=300, sigma=200, scaling_vect=None):
    if scaling_vect is None:
        output = []
        for i in range(len(length_elements_T_t)):
            output.append(scipy.stats.truncnorm.pdf(T_t[i], (a) / sigma, (b) / sigma,
                                                    loc=eval_point, scale=sigma))
    else:
        # optimize cool to optimize using numpy and vectorize.
        output = [[] for _ in range(len(length_elements_T_t))]
        for i in range(len(length_elements_T_t)):
            for j in range(length_elements_T_t[i]):
                output[i].append(1 / scaling_vect[i][j] * scipy.stats.truncnorm.pdf(T_t[i][j] / scaling_vect[i][j],
                                                                                 (a) / sigma, (b) / sigma,
                                                                                 loc=eval_point, scale=sigma))
            output[i] = np.array(output[i])
    return output


#  if important, I can generalize biweight with function beta.
#  Thus creating like 4 kernels with one function ( BETA(1), BETA(2)...)
def fct_biweight(T_t, length_elements_T_t, eval_point, a=-300, b=300, scaling_vect= None):
    if scaling_vect is None:
        output = []
        for i in range(len(length_elements_T_t)):
            xx = (T_t[i] - (a + b) / 2 - eval_point) * 2 / (b - a)
            # the correct order is eval_point - T_t,
            # bc we evaluate at eval_point but translated by T_t,
            # if kernel not symmetric a != b, then we also need to translate by the mid of them.
            xx[(xx < -1) | (xx > 1)] = 1
            output.append( 15/16 * np.power(1 - xx * xx,2) *2 / (b-a)) # kernel * scaling ; delta in my formulas
    else :
        # optimize cool to optimize using numpy and vectorize.
        output = [[] for _ in range(len(length_elements_T_t))]
        for i in range(len(length_elements_T_t)):
            for j in range(length_elements_T_t[i]):
                xx = (T_t[i][j]/scaling_vect[i][j] - (a + b) / 2 - eval_point) * 2 / (b - a) / scaling_vect[i][j]
            # the correct order is eval_point - T_t,
            # bc we evaluate at eval_point but translated by T_t,
            # if kernel not symmetric a != b, then we also need to translate by the mid of them.
                if xx < 1 or xx > 1 :
                    xx = 1
                output[i].append(15 / 16 * np.power(1 - xx * xx, 2) * 2 / (b - a))  # kernel * scaling ; delta in my formulas
            output[i] = np.array(output[i])
    return output


# # ############ test
# T_t = [np.linspace(-1000,1000,1000)]
# length_elements_T_t = [1000]
# eval_point = [-200,0,1000]
# for i in eval_point:
#     res = fct_truncnorm(T_t, length_elements_T_t, i, a=-100, b=400, sigma=300)
#     #res =  fct_biweight(T_t, length_elements_T_t, i, a=-300, b=100)
#     aplot = APlot(datax = T_t[0], datay = res[0])

