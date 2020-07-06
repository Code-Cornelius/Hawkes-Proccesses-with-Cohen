# normal libraries
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
from classes.class_kernel import *


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class Kernel_adaptive(Kernel):
    def __init__(self, fct_kernel, pilot_function_vector=None, name=' no name ', **kwargs):
        Kernel.__init__(self, fct_kernel, name, **kwargs)
        self.pilot_function_vector = pilot_function_vector

    def eval(self, T_t, eval_point, T_max):
        # if None, the Kernel adaptive is just a simple normal kernel.
        if self.pilot_function_vector is None:
            return Kernel.eval(T_t, eval_point, T_max)
        # otherwise, what I want to do is to compute the kernel classically, but with the rescale.
        else:
            # optimize cool to optimize using numpy and vectorize.
            length_elements_T_t = [len(T_t[i]) for i in range(len(T_t))]
            # ans is the kernel evaluated on the jumps
            ans = self.fct_kernel(T_t=T_t, eval_point=eval_point,
                                  length_elements_T_t=length_elements_T_t,
                                  scaling_vect=self.pilot_function_vector,
                                  **{k: self.__dict__[k] for k in self.__dict__ if
                                     k in signature(self.fct_kernel).parameters})
            # ans is a list of np arrays.
            # then I want to scale every vector.
            # The total integral should be T_max, so I multiply by T_max

            # If it isn't fct plain, then I have to scale.
            if self.fct_kernel.__name__ != 'fct_plain':
                for i in range(len(length_elements_T_t)):
                    ans[i] = ans[i] * T_max
                    # *= do not work correctly since the vectors are not the same type (int/float).
            return ans

########## test
M = 10000
INF = -1000
SUP = 1000
T_t = [np.linspace(INF,SUP,M)]
eval_point = [-200,0,400]
eval_point = [0]
my_kernel = Kernel( fct_biweight,  a=-200, b=200 )
for i in eval_point:
    res = my_kernel.eval( T_t, i, SUP-INF)
    #res =  fct_biweight(T_t, length_elements_T_t, i, a=-300, b=100)
    aplot = APlot(datax = T_t[0], datay = res[0])

T_t = [np.linspace(INF,SUP,M)]
scaling_vect = [[ (2000+x)/500 for x in list(range(M))]]
scaling_vect = [[ 2  for x in list(range(M))]]
eval_point = [-200,0,400]
eval_point = [0]
my_kernel = Kernel_adaptive( fct_biweight ,scaling_vect,  a=-200, b=200 )
for i in eval_point:
    res = my_kernel.eval( T_t, i, SUP-INF)
    #res =  fct_biweight(T_t, length_elements_T_t, i, a=-300, b=100)
    aplot = APlot(datax = T_t[0], datay = res[0])
plt.show()
