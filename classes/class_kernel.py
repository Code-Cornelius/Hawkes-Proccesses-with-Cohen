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
            #
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

        #If it isn't fct plain, then I have to scale.
        if self.fct_kernel.__name__ != 'fct_plain':
            for i in range(len(length_elements_T_t)):
                ans[i] = ans[i] * T_max  # *= do not work correctly since the vectors are not the same type (int/float).
        return ans


def fct_top_hat(T_t, length_elements_T_t, eval_point, a=-200, b=200):
    output = []
    for i in range(len(length_elements_T_t)):
        vector = np.array(T_t[i])
        output.append(1 / (2 * b - 2 * a) * \
                    (np.sign(vector - eval_point - a) +
                     np.sign(b - vector + eval_point))
                      )
    return output


def fct_plain(T_t, length_elements_T_t, eval_point):
    return [
        np.full(length_elements_T_t[i], 1)
        for i in range(len(length_elements_T_t))
    ]


def fct_truncnorm(T_t, length_elements_T_t, eval_point, a=-100, b=100, sigma=20):
    output = []
    for i in range(len(length_elements_T_t)):
        output.append(scipy.stats.truncnorm.pdf(T_t[i], (a) / sigma, (b) / sigma,
                                                loc=eval_point, scale=sigma))
    return output

#  if important, I can generalize biweight with function beta.
#  Thus creating like 4 kernels with one function ( BETA(1), BETA(2)...)
def fct_biweight(T_t, length_elements_T_t, eval_point, a = -100, b = 100):
    output = []
    for i in range(len(length_elements_T_t)):
        xx = T_t[i] - eval_point
        xx[(xx < 1) & (xx > 1)] = 0
        output.append( 15/16 * np.power(1 - xx * xx,2) )
    return output