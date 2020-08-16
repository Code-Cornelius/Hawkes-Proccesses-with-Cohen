# normal libraries
import math  # quick math functions
import numpy as np
from scipy.stats.mstats import gmean

# other files
from classes.class_kernel import *

# my libraries

np.random.seed(124)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def my_rescale_sin(value_at_each_time, G, L=None, R=None, h=3, l=0.2):
    if L is None:
        L = np.quantile(value_at_each_time, 0.15)
    print("Left boundary : ", L)
    if R is None:
        R = np.quantile(value_at_each_time, 0.95)
    print("Right boundary : ", R)
    xx = value_at_each_time - G
    # xx[ (xx < -math.pi) | (xx > math.pi) ] = math.pi

    ans = 0
    scaling1 = math.pi / (G - L)
    scaling2 = math.pi / (R - G)
    # I fix the part outside of my interest, to be the final value, h. This part corresponds to math.pi.
    # I also need the scaling by +50 given by math.pi

    # xx2 and xx3 are the cosinus, but they are different cosinus.
    # So I fix them where I don't want them to move at 0 and then I can add the two functions.
    my_xx2 = np.where((xx * scaling1 > -math.pi) & (xx * scaling1 < 0),
                      xx * scaling1, math.pi)  # left
    my_xx3 = np.where((xx * scaling2 > 0) & (xx * scaling2 < math.pi),
                      xx * scaling2, math.pi)  # right
    ans += - (h - l) / 2 * np.cos(my_xx2)
    ans += - (h - l) / 2 * np.cos(my_xx3)

    ans += l  # avoid infinite width kernel
    return ans


def classical_rescale(times, G=10., gamma=0.5):
    xx = times
    print("classical_rescale", xx)
    ans = np.power(xx / G, -gamma)
    print("classical_rescale", ans)
    return ans


def rescaling_kernel_processing(times, first_estimate,considered_param):
    # on the first entry, I get the time, on the second entry I get nu alpha or beta, then it s where in the matrix.
    #considered_param should be which parameters are important to consider.

    # ans is my vector of normes. Each value is for one time.
    ans = np.zeros(len(times))

    #times and first_estimate same length.
    # I need to pick the good parameters and rescale them accordingly.

    # the dimension of the data.
    M = len(first_estimate[0][0])
    if considered_param == ['nu']:
        total_M = M
        # I am creating a vector with 2M*M + M entries, each one is going to be scaled, and this is the parameters I am using afterwards.
        vect_of_estimators = [[] for _ in range(total_M)]
        for k in range(len(times)):
            for i in range(M):
                vect_of_estimators[i].append(first_estimate[k][0][i])
    elif considered_param == ['nu', 'alpha']:
        total_M = M*M + M
        # I am creating a vector with 2M*M + M entries, each one is going to be scaled, and this is the parameters I am using afterwards.
        vect_of_estimators = [[] for _ in range(total_M)]
        for k in range(len(times)):
            for i in range(M):
                vect_of_estimators[i].append(first_estimate[k][0][i])
                for j in range(M):
                    vect_of_estimators[M + i + j].append(first_estimate[k][1][i][j])
    else :  #if considered_param == ['nu', 'alpha', 'beta']:
        total_M = 2*M*M + M
        # I am creating a vector with 2M*M + M entries, each one is going to be scaled, and this is the parameters I am using afterwards.
        vect_of_estimators = [[] for _ in range(total_M)]
        for k in range(len(times)):
            for i in range(M):
                vect_of_estimators[i].append(first_estimate[k][0][i])
                for j in range(M):
                    vect_of_estimators[M + i + j].append(first_estimate[k][1][i][j])
                    vect_of_estimators[M + M * M + i + j].append(first_estimate[k][2][i][j])


    def rescale_min_max(vect):
        the_max = max(vect)
        the_min = min(vect)
        the_mean = classical_functions.mean_list(vect)
        ans = [ (vect[i] - the_mean) /(the_max - the_min) + 1  for i in range(len(vect))]
        return ans

    # perhaps not include the betas
    rescale_vector = [0]* (total_M)
    for i in range(total_M):
        rescale_vector[i] = rescale_min_max(vect_of_estimators[i])
    print("vect  :", vect_of_estimators)

    print("interm :", rescale_vector)
    for j in range(len(times)):
        ans[j] = np.linalg.norm([rescale_vector[i][j] for i in range(total_M)] , 2)
    print("the norms ", ans)
    # I compute the geometric mean from our estimator.
    G = gmean(ans)
    print('mean : ', G)
    scaling_factors = my_rescale_sin(ans, G=G)
    return scaling_factors


def creator_list_kernels(my_scalings, previous_scaling):
    # the kernel is taken as biweight.
    list_of_kernels = []
    for scale in my_scalings:
        new_scaling = previous_scaling / scale
        list_of_kernels.append(Kernel(fct_biweight, name="biweight", a=-new_scaling, b=new_scaling))
    return list_of_kernels

############ test adaptive window
# T_t = [np.linspace(0.1,100,10000)]
# G = 10.
# #T_t = [np.random.randint(0,6*G, 20)]
# eval_point = [0]
# for i in eval_point:
#     min = np.quantile(T_t, 0.02)
#     max = np.quantile(T_t, 0.75)
#     res = test_geom_kern(T_t, G, min = min, max = max)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = T_t[0], yy = res[0])
#     aplot.plot_vertical_line(G, np.linspace(-5,105, 1000), nb_ax=0, dict_plot_param={'color':'k', 'linestyle':'--', 'markersize':0, 'linewidth':2, 'label':'geom. mean'})
#     aplot.plot_vertical_line(min, np.linspace(-5, 105, 1000), nb_ax=0,
#                              dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2, 'label':'lower bound'})
#     aplot.plot_vertical_line(max, np.linspace(-5, 105, 1000), nb_ax=0,
#                              dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2, 'label':'upper bound'})
#     aplot.set_dict_fig(0, {'title':'Adaptive scaling for Adaptive Window Width','xlabel':'Value', 'ylabel':'Scaling'})
#     aplot.show_legend()
#
# eval_point = [0]
# for i in eval_point:
#     res = test_normal_kernel(T_t, G, gamma = 0.5)
#     aplot = APlot(how = (1,1))
#     aplot.uni_plot(nb_ax = 0, xx = T_t[0], yy = res[0])
#     aplot.plot_vertical_line(G, np.linspace(-1,10, 1000), nb_ax=0, dict_plot_param={'color':'k', 'linestyle':'--', 'markersize':0, 'linewidth':2, 'label':'geom. mean'})
#     aplot.set_dict_fig(0, {'title':'Adaptive scaling for Adaptive Window Width','xlabel':'Value', 'ylabel':'Scaling'})
#     aplot.show_legend()
