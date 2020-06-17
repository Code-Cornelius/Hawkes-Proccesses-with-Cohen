import cmath
import numpy as np


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
    if time / T_max * 2 * cmath.pi * 2.25 < 2 * cmath.pi * 1.75:
        return base_value + a * cmath.cos(time / T_max * 2 * cmath.pi * 2.25) * cmath.cos(
            time / T_max * 2 * cmath.pi * 2.25)
    else:
        return base_value
