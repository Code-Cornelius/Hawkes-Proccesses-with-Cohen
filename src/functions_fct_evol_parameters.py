import math
import numpy as np


# all of those functions are simply function that returns one element.

#todo add to the other movements :
def constant_parameter(time, constant, T_max = 0):
    return constant

# a is already divided by t_max, so just put of how much you want to grow
def linear_growth(time, a, b, T_max):  # ax + b
    return a / T_max * time + b


# when jump should be a %
def one_jump(time, when_jump, original_value, new_value, T_max):
    return original_value + new_value * np.heaviside(time - T_max * when_jump, 1)


# when jump should be a %
# ax+b until the when_jump, where it comes down to base value.
def moutain_jump(time, when_jump, a, b, base_value, T_max):
    if time < when_jump * T_max:
        return linear_growth(time, a, b, T_max)
    else:
        return base_value


#
def periodic_stop(time, T_max, a, base_value):  # need longer realisation like 80 mini_T
    if time / T_max * 2 * math.pi * 2.25 < 2 * math.pi * 1.75:
        return base_value + a * math.cos(time / T_max * 2 * math.pi * 2.25) * math.cos(
            time / T_max * 2 * math.pi * 2.25)
    else:
        return base_value


# import numpy as np
# from plot_functions import *
#
#
# list_1 = [constant_parameter, linear_growth, one_jump, moutain_jump, periodic_stop]
# list_2 = [ {'constant' : 5}, {'a': 2,'b':4}, {'when_jump': 0.4, 'original_value':2, 'new_value':3},
#                                           {'when_jump':0.7, 'a':2, 'b':1.5, 'base_value':0.5},
#                                           {'base_value': 3,'a':1}]
#
# T_max = 1000
# xx = np.linspace(0,T_max,100000)
# for fct,param in zip(list_1, list_2):
#     fct = np.vectorize(fct)
#     yy = fct(xx, T_max = T_max, **param)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = xx, yy = yy, dict_plot_param = {"color" : "blue", "markersize": 0, "linewidth" : 2})
#     aplot.set_dict_fig(0, {'title': '', 'xlabel':'', 'ylabel':''})
#
# plt.show()