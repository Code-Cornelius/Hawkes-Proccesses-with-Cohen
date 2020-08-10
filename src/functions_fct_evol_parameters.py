import math
import numpy as np
from functools import partial

# all of those functions are simply function that returns one element.

# be sure that the behaviour after T_max is ok.
# Sometimes, I need the function for values slightly bigger than T_max (last event).


#todo add to the other movements :
import functions_general_for_Hawkes


def constant_parameter(time, constant, T_max = 0, time_burn_in = 0):
    return constant

# a is already divided by t_max, so just put of how much you want to grow
def linear_growth(time, a, b, T_max, time_burn_in):  # ax + b
    return a / (T_max+time_burn_in) * time + b


# when jump should be a %
def one_jump(time, when_jump, original_value, new_value, T_max, time_burn_in):
    return original_value + new_value * np.heaviside(time - time_burn_in - T_max * when_jump, 1)


# when jump should be a %
# ax+b until the when_jump, where it comes down to base value.
def moutain_jump(time, when_jump, a, b, base_value, T_max, time_burn_in):
    if time < when_jump * T_max + time_burn_in:
        return linear_growth(time, a, b, T_max, time_burn_in)
    else:
        return base_value


def periodic_stop(time, T_max, a, base_value, time_burn_in):  # need longer realisation like 80 mini_T
    if time / (T_max + time_burn_in) * 2 * math.pi * 2.25 < 2 * math.pi * 1.75:
        return base_value + a * math.cos(time / (T_max+time_burn_in) * 2 * math.pi * 2.25) * math.cos(
            time / (T_max+time_burn_in) * 2 * math.pi * 2.25)
    else:
        return base_value



def update_functions(case, PARAMETERS):
    MU,ALPHA,BETA = PARAMETERS
    M = len(MU)
    the_update_functions = functions_general_for_Hawkes.multi_list_generator(M)

    # for 7500 jumps, do 210 with first sets of param dim 1
    if case == 0:
        for i in range(M):
            value = MU[i]
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, i: constant_parameter(time, value, T_max=T_max, time_burn_in = time_burn_in), i = i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i,j: constant_parameter(time, ALPHA[i, j], T_max=T_max, time_burn_in= time_burn_in), i = i, j = j)
                the_update_functions[2][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i,j: constant_parameter(time, BETA[i, j], T_max=T_max, time_burn_in= time_burn_in), i = i, j = j)

    # for 7500 jumps, do 60 with first sets of param dim 1
    if case == 1:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, i: linear_growth(time, 1.5 * MU[i], MU[i]/2, T_max, time_burn_in= time_burn_in), i = i)

            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: linear_growth(time, BETA[i, j]*0.8 - ALPHA[i, j], ALPHA[i, j], T_max, time_burn_in= time_burn_in), i = i, j = j) # it goes up to BETA 90%

                # the_update_functions[2][i][j] = \
                #     lambda time, T_max, time_burn_in: functions_fct_evol_parameters.linear_growth(time, 3, BETA[i, j], T_max, time_burn_in = time_burn_in)
                the_update_functions[2][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: constant_parameter(time, BETA[i, j], T_max=T_max, time_burn_in= time_burn_in), i = i, j = j)

    # for 7500 jumps, do 100 with first sets of param dim 1
    elif case == 2:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, i: one_jump(time, 0.7, MU[i]/3, 2*MU[i], T_max, time_burn_in= time_burn_in), i = i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: one_jump(time, 0.4, ALPHA[i, j]/3, BETA[i, j]*0.7 - ALPHA[i, j]/3,
                                                                               T_max , time_burn_in = time_burn_in), i = i, j = j)
                # the_update_functions[2][i][j] = \
                #     lambda time, T_max, time_burn_in: functions_fct_evol_parameters.one_jump(time, 0.4, BETA[i, j], BETA[i, j], T_max, time_burn_in= time_burn_in)
                the_update_functions[2][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: constant_parameter(time, BETA[i, j], T_max=T_max, time_burn_in = time_burn_in), i = i, j = j)

    # for 7500 jumps, do 100 with first sets of param dim 1
    elif case == 3:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, i: moutain_jump(time, when_jump=0.7, a=MU[i], b=MU[i],
                                                                               base_value=MU[i] * 1., T_max=T_max, time_burn_in= time_burn_in), i = i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: moutain_jump(time, when_jump=0.5, a= 2* (BETA[i, j]*0.8 - ALPHA[i, j]),
                                                                                   b=ALPHA[i, j],
                                                                                   base_value=ALPHA[i, j] / 2,
                                                                                   T_max=T_max, time_burn_in= time_burn_in), i = i, j = j)
                # the_update_functions[2][i][j] = \
                #     lambda time, T_max, time_burn_in: functions_fct_evol_parameters.moutain_jump(time, when_jump=0.7, a=1.8,
                #                                                                    b=BETA[i, j],
                #                                                                    base_value=BETA[i, j] / 1.5,
                #                                                                    T_max=T_max, time_burn_in= time_burn_in)
                the_update_functions[2][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: constant_parameter(time, BETA[i, j],
                                                                                                       T_max=T_max,
                                                                                                       time_burn_in=time_burn_in), i = i, j = j)
    # for 7500 jumps, do 60 with first sets of param dim 1
    elif case == 4:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, i: periodic_stop(time, T_max, MU[i], 0.2, time_burn_in=time_burn_in), i = i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: periodic_stop(time, T_max, BETA[i, j]*0.9 - ALPHA[i, j]/2, ALPHA[i, j] / 2, time_burn_in=time_burn_in), i = i, j = j)
                # the_update_functions[2][i][j] = \
                #     lambda time, T_max, time_burn_in: functions_fct_evol_parameters.periodic_stop(time, T_max, BETA[i, j], 2.5, time_burn_in=time_burn_in)
                the_update_functions[2][i][j] = \
                    partial(lambda time, T_max, time_burn_in, i, j: constant_parameter(time, BETA[i, j],
                                                                         T_max=T_max,
                                                                         time_burn_in=time_burn_in), i = i, j = j)
    return the_update_functions




# import numpy as np
# from plot_functions import *
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
#     yy = fct(xx, T_max = T_max, time_burn_in = 0, **param)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = xx, yy = yy, dict_plot_param = {"color" : "blue", "markersize": 0, "linewidth" : 2})
#     aplot.set_dict_fig(0, {'title': '', 'xlabel':'', 'ylabel':''})
#
# plt.show()