import numpy as np
import scipy.stats

from inspect import signature #used in the method eval of the class




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

class Kernel:
    # kernel is a class of objects, where using eval evaluates the function given as parameter
    # the evaluation gives back a list of np.array
    # the function should hand in the list of np.arrays non scaled.

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

    def eval(self, T_t, eval_point):
        # optimize cool to optimize using numpy and vectorize.
        length_elements_T_t = [len(T_t[i]) for i in range(len(T_t))]
        ans = self.fct_kernel(T_t=T_t, eval_point=eval_point, length_elements_T_t=length_elements_T_t,
                              **{k: self.__dict__[k] for k in self.__dict__ if
                                 k in signature(self.fct_kernel).parameters})
        for i in range(len(length_elements_T_t)):
            value = np.sum(ans[i])
            scale_factor = length_elements_T_t[i]
            factor = scale_factor / value
            ans[i] = ans[i] * factor  # *= do not work correctly since the vectors are not the same type.
        return ans


def fct_top_hat(T_t, length_elements_T_t, eval_point, a=-200, b=200):
    output = [[] for _ in range(len(length_elements_T_t))]
    for i in range(len(length_elements_T_t)):
        vector = np.array(T_t[i])
        output[i] = 1 / (2 * b - 2 * a) * \
                    (np.sign(vector - eval_point - a) +
                     np.sign(b - vector + eval_point))
    return output


def fct_plain(T_t, length_elements_T_t, eval_point):
    return [
        np.full(length_elements_T_t[i], 1)
        for i in range(len(length_elements_T_t))
    ]


def fct_truncnorm(T_t, length_elements_T_t, eval_point, a=-100, b=100, sigma=20):
    output = [[] for _ in range(len(length_elements_T_t))]
    for i in range(len(length_elements_T_t)):
        vector = np.array(T_t[i])
        output[i] = scipy.stats.truncnorm.pdf(T_t[i], (a) / sigma, (b) / sigma, loc=eval_point, scale=sigma)
    return output
