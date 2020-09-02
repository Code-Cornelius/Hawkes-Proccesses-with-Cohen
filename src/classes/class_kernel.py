##### normal libraries
from inspect import signature  # used in the method eval of the class

import numpy as np
import scipy.stats  # functions of statistics
from library_classes.estimators.graphs.class_graph_estimator import *
##### my libraries
from library_functions.tools import classical_functions_integration

##### other files

np.random.seed(124)


# section ######################################################################
#  #############################################################################
# some information


# -------------------------------------------------------------------------------------------------------
# list of the possible kernels:
# fct_top_hat
# fct_plain
# fct_truncnorm
# fct_biweight
#
#

# the functions are correct, they scale and shift the way it is supposed.
# However they are written in the following way : f_t(t_i) = K( t_i - t )

# example of kernels:
# list_of_kernels =
#           [Kernel(fct_top_hat, name="wide top hat", a=-450, b=450),
#            Kernel(fct_top_hat, name="normal top hat", a=-200, b=200),
#            Kernel(fct_truncnorm, name="wide truncnorm", a=-500, b=500, sigma=350),
#            Kernel(fct_truncnorm, name="normal truncnorm", a=-350, b=350, sigma=250)]
# -------------------------------------------------------------------------------------------------------
# the functions only work for positive time. If one input negative times, it messes up the orientation.


# section ######################################################################
#  #############################################################################
# class
class Kernel:
    # kernel is a class of objects, where using eval evaluates the function given as parameter
    # the evaluation gives back a list of np.array
    # the function should hand in the list of np.arrays non scaled.
    # list of arrays allow me to perform computations on each faster.

    # I also chose to not give T_t to the kernel in order to
    # make it a function of the T_t (and of where one evaluates the kernel),
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
        return repr(self._fct_kernel)

    def eval(self, T_t, eval_point, T_max):
        length_elements_T_t = [len(T_t[i]) for i in range(len(T_t))]
        # ans is the kernel evaluated on the jumps
        ans = self._fct_kernel(T_t=T_t, eval_point=eval_point, length_elements_T_t=length_elements_T_t,
                               **{k: self.__dict__[k] for k in self.__dict__ if
                                  k in signature(self._fct_kernel).parameters})
        # ans is a list of np arrays. It is normalized such that it is a kernel.
        # then I want to scale every vector.
        # The total integral should be T_max, so I multiply by T_max

        # If it isn't fct plain, then I have to scale.
        if self._fct_kernel.__name__ != 'fct_plain':

            # I want to rescale the results for the kernels that are not covering seen part. For that reason,
            # I compute the integral of the kernel, and scale accordingly.
            # todo the linspace shouldn't be 0 T_max but the point at which I start the simulation...
            tt_integral = [np.linspace(0, T_max, 10000)]
            yy = self._fct_kernel(T_t=tt_integral, eval_point=eval_point, length_elements_T_t=[1],
                                  **{k: self.__dict__[k] for k in self.__dict__ if
                                     k in signature(self._fct_kernel).parameters})
            integral = classical_functions_integration.trapeze_int(tt_integral[0],
                                                                   yy[
                                                                       0])  # yy[0] bc function gives back a list of arrays.

            for i in range(len(length_elements_T_t)):
                ans[i] = ans[i] / integral * T_max
                # *= do not work correctly since the vectors are not the same type (int/float).
                # I also divide by the sum, the vector is normalized, however,
                # possibly we're on the edge and we need to take that into account.
                # print(
                #     f"inside kernel debug, that's my "
                #     f"integral : {np.sum(ans[i][:-1]) * T_max / (len(ans[i]) - 1)}. Name : {self.fct_kernel.__name__}.")
        return ans

    # section ######################################################################
    #  #############################################################################
    # getters setters
    @property
    def fct_kernel(self):
        return self._fct_kernel

    @fct_kernel.setter
    def fct_kernel(self, new_fct_kernel):
        self._fct_kernel = new_fct_kernel

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if isinstance(new_name, str):
            self._name = new_name
        else:
            raise Error_type_setter(f'Argument is not an string.')


def fct_top_hat(T_t, length_elements_T_t, eval_point, a=-200, b=200):
    output = []
    for i in range(len(length_elements_T_t)):
        vector = np.array(T_t[i])
        # -1 if x < 0, 0 if x==0, 1 if x > 0.
        output.append(1 / (2 * (b - a)) *
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
        output.append(scipy.stats.truncnorm.pdf(np.array(T_t[i]), a / sigma, b / sigma,
                                                loc=eval_point, scale=sigma))
    return output


def fct_truncnorm_test(T_t, length_elements_T_t, eval_point, a=-300, b=300, sigma=200):
    output = []
    i = 0  # for output[i] after, but there shouldn't be any problem.
    for i in range(len(length_elements_T_t)):
        output.append(2 * scipy.stats.truncnorm.pdf(np.array(T_t[i]), a / sigma, b / sigma,
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

# section ######################################################################
#  #############################################################################
# test


# T_t = [np.linspace(0,2000,10000)]
# aplot = APlot(how = (1,1))
# aplot.set_dict_fig(0, {'title':"", 'xlabel':"", 'ylabel':""})
#
#
# color = plt.cm.Dark2.colors
# for fct,c in zip([fct_truncnorm],color):
#     my_kernel = Kernel(fct, a=-500, b=500)
#     length_elements_T_t = [10000]
#     eval_point = [0, 100, 250, 1000,1750,1900, 2000]
#     for i_point in eval_point:
#         res = my_kernel.eval( T_t, i_point, 2000)
#         aplot.uni_plot(nb_ax=0, xx=T_t[0], yy=res[0], dict_plot_param={
#         "color":c, "label":str(fct.__name__) + " evaluated at " + str(i_point),
#                                                                        "markersize" : 0, "linewidth":2})
# aplot.show_legend()
#
#
#
# plt.show()
