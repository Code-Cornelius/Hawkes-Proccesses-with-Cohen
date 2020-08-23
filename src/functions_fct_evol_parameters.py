import math
from functools import partial

import numpy as np

import functions_general_for_Hawkes


# all of those functions are simply function that returns one element.
# be sure that the behaviour for values bigger than T_max is ok.
# Sometimes, I need the function for values slightly bigger than T_max (last event).


def constant_parameter(constant, *args, **kwargs):
    return constant


# a is already divided by t_max, so just put of how much you want to grow
def linear_growth(time, a, b, T_max, time_burn_in):  # ax + b
    return a / (T_max + time_burn_in) * time + b


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
        return base_value + a * math.cos(time / (T_max + time_burn_in) * 2 * math.pi * 2.25) * math.cos(
            time / (T_max + time_burn_in) * 2 * math.pi * 2.25)
    else:
        return base_value


def update_functions(case, PARAMETERS):
    MU, ALPHA, BETA = PARAMETERS
    M = len(MU)
    the_update_functions = functions_general_for_Hawkes.multi_list_generator(M)

    # for 7500 jumps, do 120 with first sets of param dim 1
    if case == 0:
        for i in range(M):
            value = MU[i]
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, index_1: constant_parameter(time=time, constant=value,
                                                                                      T_max=T_max,
                                                                                      time_burn_in=time_burn_in), index_1=i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(
                        time=time,
                        constant=ALPHA[
                            index_1, index_2],
                        T_max=T_max,
                        time_burn_in=time_burn_in),
                            i=i,
                            j=j)
                the_update_functions[2][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(time=time, constant=BETA[
                            index_1, index_2], T_max=T_max, time_burn_in=time_burn_in),
                        index_1=i,
                        index_2=j)

    # for 7500 jumps, do 70 with first sets of param dim 1
    if case == 1:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(
                    lambda time, T_max, time_burn_in, index_1: linear_growth(time, 1.5 * MU[index_1], MU[index_1] / 2,
                                                                             T_max, time_burn_in=time_burn_in), index_1=i)

            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, index_1, index_2: linear_growth(
                        time, BETA[index_1, index_2] * 0.8 - ALPHA[index_1, index_2], ALPHA[index_1, index_2], T_max,
                        time_burn_in=time_burn_in),
                            index_1=i, index_2=j)  # it goes up to BETA 90%

                the_update_functions[2][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(time=time, constant=BETA[
                            index_1, index_2], T_max=T_max, time_burn_in=time_burn_in), index_1=i, index_2=j)

    # for 7500 jumps, do 80 with first sets of param dim 1
    elif case == 2:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, index_1: one_jump(time, 0.7, MU[index_1], 1.7 * MU[index_1],
                                                                            T_max, time_burn_in=time_burn_in), index_1=i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(lambda time, T_max, time_burn_in, index_1, index_2: one_jump(time, 0.4,
                                                                                         BETA[index_1, index_2] * 0.7,
                                                                                         -0.5, T_max,
                                                                                         time_burn_in=time_burn_in),
                            index_1=i, index_2=j)
                the_update_functions[2][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(time=time, constant=BETA[
                            index_1, index_2], T_max=T_max, time_burn_in=time_burn_in), index_1=i, index_2=j)

    # for 7500 jumps, do 100 with first sets of param dim 1
    elif case == 3:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, index_1: moutain_jump(time, when_jump=0.7, a=MU[index_1],
                                                                                b=MU[index_1],
                                                                                base_value=MU[index_1] * 1.,
                                                                                T_max=T_max, time_burn_in=time_burn_in),
                        index_1=i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: moutain_jump(
                            time, when_jump=0.5,
                            a=2 * (BETA[index_1, index_2] * 0.8 -
                                   ALPHA[index_1, index_2]),
                            b=ALPHA[index_1, index_2],
                            base_value=ALPHA[index_1, index_2] / 1.2,
                            T_max=T_max,
                            time_burn_in=time_burn_in), index_1=i, index_2=j)
                the_update_functions[2][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(time=time, constant=BETA[
                            index_1, index_2], T_max=T_max, time_burn_in=time_burn_in), index_1=i, index_2=j)
    # for 7500 jumps, do 65 with first sets of param dim 1
    elif case == 4:
        for i in range(M):
            the_update_functions[0][i] = \
                partial(lambda time, T_max, time_burn_in, index_1: periodic_stop(time, T_max, MU[index_1], 0.2,
                                                                                 time_burn_in=time_burn_in), i=i)
            for j in range(M):
                the_update_functions[1][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: periodic_stop(
                            time, T_max, BETA[index_1, index_2] * 0.8 - ALPHA[index_1, index_2],
                            ALPHA[index_1, index_2], time_burn_in=time_burn_in), index_1=i, index_2=j)
                the_update_functions[2][i][j] = \
                    partial(
                        lambda time, T_max, time_burn_in, index_1, index_2: constant_parameter(time=time, constant=BETA[
                            index_1, index_2], T_max=T_max, time_burn_in=time_burn_in), index_1 = i, index_2=j)
    return the_update_functions
