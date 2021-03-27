# normal libraries
import math  # quick math functions

from scipy.stats.mstats import gmean

# my libraries
from library_functions.tools import classical_functions_vectors
from library_errors.Error_not_allowed_input import Error_not_allowed_input
# other files
from classes.class_kernel import *

np.random.seed(124)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def my_rescale_sin(value_at_each_time, L=0.02, R=0.98, h=2.5, l=0.2 / 2, silent=True):
    if any(value_at_each_time != 0):
        # I compute the geometric mean from our estimator.
        G = gmean(value_at_each_time)

    else:  # G == 0, it happens if no norm computed.
        # then it has to return 0.01 such that it widen all the kernels.
        return np.full(len(value_at_each_time), 0.01)

    L_quant = np.quantile(value_at_each_time, L)
    R_quant = np.quantile(value_at_each_time, R)

    if not L_quant < G < R_quant:
        raise Error_not_allowed_input("L < G < R for the well definiteness of the function.")

    if not silent:
        print("Left boundary : ", L_quant)
    if not silent:
        print("Right boundary : ", R_quant)

    xx = value_at_each_time - G

    ans = 0
    scaling1 = math.pi / (G - L_quant)
    scaling2 = math.pi / (R_quant - G)
    # I fix the part outside of my interest, to be the final value, h.
    # This part corresponds to math.pi.
    # I also need the scaling by +h/2 given by math.pi

    # xx2 and xx3 are the cosinus, but they are different cosinus.
    # So I fix them where I don't want them to move at 0 and then I can add the two functions.
    my_xx2 = np.where((xx * scaling1 > -math.pi) & (xx * scaling1 < 0),
                      xx * scaling1, math.pi)  # left
    my_xx3 = np.where((xx * scaling2 > 0) & (xx * scaling2 < math.pi),
                      xx * scaling2, math.pi)  # right
    ans += - (h - l) / 2 * np.cos(my_xx2)
    ans += - (h - l) / 2 * np.cos(my_xx3)

    ans += l  # avoid infinite width kernel, with a minimal value.
    return ans


def AKDE_scaling(times, G=10., gamma=0.5):
    xx = times
    print("classical_rescale", xx)
    ans = np.power(xx / G, -gamma)
    print("classical_rescale", ans)
    return ans


def rescale_min_max(my_vect):
    the_max = max(my_vect)
    the_min = min(my_vect)
    the_mean = classical_functions_vectors.mean_list(my_vect)
    ans = [(my_vect[i] - the_mean) / (the_max - the_min) + 1 for i in range(len(my_vect))]
    return ans


def check_if_any_evolution(vector, tol):
    """ if all values of the vector are inside the tube mean +/- tol, return false.
    Args:
        vector:
        tol:

    Returns:

    """
    the_mean = classical_functions_vectors.mean_list(vector)
    if all(element < the_mean * (1 + tol) for element in vector) and all(
            element > the_mean * (1 - tol) for element in vector):
        return False
    return True


def rescaling_kernel_processing(times, first_estimate, considered_param, L, R, h, l, tol=0., silent=True):
    # on the first entry, I get the time, on the second entry I get nu alpha or beta,
    # then it s where in the matrix.
    # considered_param should be which parameters are important to consider.

    # norm_over_the_time is my vector of norms. Each value is for one time.
    norm_over_the_time = np.zeros(len(times))

    # times and first_estimate same length.
    # I need to pick the good parameters and rescale them accordingly.

    # the dimension of the data.
    M = len(first_estimate[0][0])
    total_M = 2 * M * M + M
    include_estimation = [False] * total_M
    # I am creating a vector with 2M*M + M entries, each one is going to be scaled,
    # and this is the parameters I am using afterwards.
    vect_of_estimators = [[] for _ in range(total_M)]
    for k in range(len(times)):
        for i in range(M):
            vect_of_estimators[i].append(first_estimate[k][0][i])
            for j in range(M):
                vect_of_estimators[M + i + j].append(first_estimate[k][1][i][j])
                vect_of_estimators[M + M * M + i + j].append(first_estimate[k][2][i][j])

    for i in range(total_M):
        # check the parameters I need to check.
        if i < M and 'nu' in considered_param:
            include_estimation[i] = True
        elif i < M + M * M and 'alpha' in considered_param:
            include_estimation[i] = True
        elif 'beta' in considered_param:
            include_estimation[i] = True

        if include_estimation[i]:
            if not check_if_any_evolution(vect_of_estimators[i], tol):  # we don't keep the True
                include_estimation[i] = False
    if not silent:
        print("which dim to include for norm : (nu,alpha,beta);", include_estimation)

    rescale_vector = []
    for i in range(total_M):
        if include_estimation[i]:
            rescale_vector.append(rescale_min_max(vect_of_estimators[i]))

    for j in range(len(times)):
        norm_over_the_time[j] = np.linalg.norm([rescale_vector[i][j] for i in range(len(rescale_vector))], 2)
    if not silent:
        print("vect  :", vect_of_estimators)
        print("the norms ", norm_over_the_time)
    scaling_factors = my_rescale_sin(norm_over_the_time, L=L, R=R, h=h, l=l, silent=silent)
    return scaling_factors


def creator_list_kernels(my_scalings, list_previous_half_width, first_width=None):
    #todo would be interesting to add name to the function such that one can do iterative adaptive scaling.

    # we want that both inputs my_scalings and list_previous_half_width are the same size

    # the kernel is taken as biweight.
    list_half_width = []
    list_of_kernels = []

    if not len(my_scalings) == len(list_previous_half_width):
        raise Error_not_allowed_input("Both lists need to be the same size.")

    if first_width is None:
        first_width = list_previous_half_width[0]

    for half_width, scale in zip(list_previous_half_width, my_scalings):
        new_scaling = half_width / scale
        list_half_width.append(new_scaling)
        list_of_kernels.append(
            Kernel(fct_biweight,
                   name=f"Adaptive Biweight with first width {2 * first_width}.",
                   a=-new_scaling, b=new_scaling)
        )
    return list_half_width, list_of_kernels


def creator_kernels_adaptive(my_estimator_mean_dict, Times, considered_param, list_previous_half_width, L, R, h, l,
                             tol=0.1, silent=True):
    # todo check times match

    #list half width:  sequence of all the half width.
    #list_of_kernels :  sequence of all the new kernels.

    # by looking at the previous estimation, we deduce the scaling
    # for that I take back the estimate
    # there is a problem of data compatibility, so I put the keys as integers,
    # assuming that there is no estimation on the same integer.

    # tolerance is by how much the dimension has to move in order to consider
    # that it is worth updating wrt to it. Tol is % of base value.

    my_estimator_dict = my_estimator_mean_dict.estimator_mean(
        separator='time estimation')  # take back the value of the estimation at a given time.
    my_estimator_dict = {int(key): my_estimator_dict[key] for key in my_estimator_dict.keys()}
    list_of_estimation = []
    # mean returns a dict, so I create my list of list:
    for a_time in Times:
        a_time = int(a_time)
        list_of_estimation.append(my_estimator_dict[a_time])

    my_scaling = rescaling_kernel_processing(
        times=Times, first_estimate=list_of_estimation,
        considered_param=considered_param, tol=tol, L=L, R=R, h=h, l=l, silent=silent)
    if not silent:
        print('the scaling : ', my_scaling)
    # the kernel is taken as biweight.
    list_half_width, list_of_kernels = creator_list_kernels(my_scalings=my_scaling, list_previous_half_width=list_previous_half_width)
    return list_half_width, list_of_kernels
