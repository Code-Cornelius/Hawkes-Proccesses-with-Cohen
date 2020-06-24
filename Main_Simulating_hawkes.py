##### normal libraries
import numpy as np
import statistics as stat
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time


##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions

##### other files
import functions_MLE
import class_kernel
from class_hawkes_process import *
from class_estimator_hawkes import *
from class_graph_hawkes import *
import functions_general_for_Hawkes
import functions_change_point_analysis
import functions_fct_evol_parameters





##########################################
################parameters################
##########################################

# timing
# 1D
T0, mini_T = 0, 35 # 50 jumps for my uni variate stuff
# 2D
T0, mini_T = 0, 120

# so here I should have around 500 jumps.
#T = 10 * mini_T
# 2000 JUMPS
T = 200 * mini_T

####################################################################### TIME
# number of max jump
nb_of_sim, M_PREC = 50000,200000
M_PREC += 1
# a good precision is 500*(T-T0)


tt = np.linspace(T0, T, M_PREC, endpoint=True)
'''
ALPHA = [[0.5, 0.3, 0.3],
         [0.3, 0.5, 0.3],
         [0.3, 0.3, 0.5]]
BETA = [[100, 5, 5],
        [10, 100, 5],
        [1.5, 5, 100]]
MU = [0.3, 0.3, 0.3]
#'''
'''
ALPHA = [[0.5, 0.3, 0.3, 0, 0, 0],
         [0.3, 0.5, 0.3, 0, 0, 0],
         [0.3, 0.3, 0.5, 0, 0, 0],
         [0, 0, 0, 1, 0.1, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0.1, 1]]
BETA = [[100, 5, 5, 0, 0, 0],
        [10, 100, 5, 0, 0, 0],
        [1.5, 5, 100, 0, 0, 0],
        [0, 0, 0, 4, 2, 0],
        [0, 0, 0, 0, 10, 0],
        [0, 0, 0, 0, 3, 3]]
MU = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
'''
#'''
ALPHA = [[0.4, 0.1], [0.1, 0.4]]
BETA = [[1.2, 0.6], [0.6, 1.2]]
MU = [0.2, 0.2]

# ALPHA = [[2, 0], [0, 3]]
# BETA = [[20, 20], [10, 10]]
# MU = [0.7, 0.3]
#'''
'''
ALPHA = [[1.75]]
BETA = [[2]]
MU = [0.2]
#'''
'''
ALPHA = [[0.6]]
BETA = [[1.2]]
MU = [1.2]
#'''
ALPHA, BETA, MU = np.array(ALPHA, dtype=np.float), np.array(BETA, dtype=np.float), np.array(MU, dtype=np.float) # I precise the type because he might think the np.array is int type.
PARAMETERS = [MU.copy(), ALPHA.copy(), BETA.copy()]

print("ALPHA : \n", ALPHA)
print("BETA : \n", BETA)
print('MU : \n', MU)
print("=" * 78)
print("=" * 78)
print("=" * 78)
np.random.seed(124)

HAWKSY = Hawkes_process(tt, PARAMETERS)

estimator_multi = Estimator_Hawkes()
################################################
# plot
plot = False
################################################
if plot:
    intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool = True, silent = True)
    HAWKSY.plot_hawkes(time_real, intensity, name = "EXACT_HAWKES")
    plt.show()


# graph = Graph.from_path(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_test.csv',
#                         PARAMETERS)
# print(graph.estimator)
# print(
#     functions_general_for_Hawkes.mean_HP(graph.estimator)
# )



#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
################################################
# simulation
silent = False
test_mode = True
################################################
################################################
if test_mode :
    nb_of_guesses, T = 10, 30 * mini_T
else:
    nb_of_guesses, T = 50, 100 * mini_T
tt = np.linspace(T0, T, M_PREC, endpoint=True)
################################################
################################################
the_update_functions = functions_general_for_Hawkes.multi_list_generator(HAWKSY.M)
# choice of case study
#########
case = 3
#########
if case == 1:
    for i in range(HAWKSY.M):
        the_update_functions[0][i] = lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 0.1, MU[i], T_max)
        for j in range(HAWKSY.M):
            the_update_functions[1][i][j] = lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 2, ALPHA[i,j], T_max)
            the_update_functions[2][i][j] = lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 3, BETA[i,j], T_max)

elif case == 2:
    for i in range(HAWKSY.M):
        the_update_functions[0][i] = lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.1, MU[i], 0, T_max)
        for j in range(HAWKSY.M):
            the_update_functions[1][i][j] = lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.7, ALPHA[i, j], ALPHA[i, j], T_max)
            the_update_functions[2][i][j] = lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.4, BETA[i, j], BETA[i, j], T_max)

elif case == 3:
    for i in range(HAWKSY.M):
        the_update_functions[0][i] = lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump = 0.7, a = 0, b = MU[i], base_value =  MU[i] * 1.5, T_max = T_max)
        for j in range(HAWKSY.M):
            the_update_functions[1][i][j] = lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump = 0.4, a =1.4, b =ALPHA[i, j], base_value = ALPHA[i, j] / 2,
                                                                     T_max =T_max)
            the_update_functions[2][i][j] = lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump = 0.7, a = 1.8, b =BETA[i, j], base_value = BETA[i, j] / 1.5,
                                                                     T_max =T_max)

elif case == 4:
    for i in range(HAWKSY.M):
        the_update_functions[0][i] = lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, MU[i], 0.2)
        for j in range(HAWKSY.M):
            the_update_functions[1][i][j] = lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, ALPHA[i, j], 1)
            the_update_functions[2][i][j] = lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, BETA[i, j], 2.5)


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
print("\n~~~~~Computations.~~~~~\n")
do = False ###################################### SIMPLE UNIQUE
if do:
    intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool = False, silent = silent)
    print( functions_MLE.call_newton_raph_MLE_opt(time_real, T, silent = silent) )



#-----------------------------------------------------------------------------------------------
do = False ###################################### SIMPLE MULTI
if do:
    estimator_multi = Estimator_Hawkes()
    functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_multi, T, nb_of_guesses, silent = silent)
    GRAPH_multi = Graph_Hawkes(estimator_multi, the_update_functions)
    GRAPH_multi.histogram_of_realisations_of_estimator()

    estimator_multi.to_csv(r'/home/bianca/est/estimators.csv', index=False,
                           header=True)

# -----------------------------------------------------------------------------------------------
do = False  ###################################### OVER THE TIME ESTIMATION, DIFFERENT KERNELS
nb_of_times = 3
if do:
    HAWKSY = Hawkes_process(tt, PARAMETERS)
    # I create here the array. It is quite hard because I want a list of size size*size*3 where all elements can be change however I want. Other ways lead dependant vectors.

    # Here I change the parameters over time.
    estimator_kernel = Estimator_Hawkes()
    estimator_kernel.append(pd.read_csv(r'/home/bianca/est/estimators_new.csv'))
    # list_of_kernels = [Kernel(fct_truncnorm, name="my truncnorm", a=-350, b=350, sigma=300),
    #                    Kernel(fct_truncnorm, name="large truncnorm", a=-500, b=500, sigma=300)]
    #                    # Kernel(fct_truncnorm, name="large, high truncnorm", a=-500, b=500, sigma=450)]
    # Times = np.linspace(0.1 * T, 0.9 * T, nb_of_times)
    #
    # count_kernels = 0;
    # count_times = 0
    # for time in Times:
    #     count_kernels = 0
    #     count_times += 1
    #     HAWKSY.update_coef(time, the_update_functions, T_max=T)
    #     print(HAWKSY)
    #     for kernel in list_of_kernels:
    #         count_kernels += 1
    #         print("=" * 78)
    #         print("Time : {} out of : {}. Kernel : {} out of : {}.".format(count_times, len(Times), count_kernels,
    #                                                                        len(list_of_kernels)))
    #         functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max=T, nb_of_guesses=nb_of_guesses,
    #                                                     kernel_weight=kernel, time_estimation=time, silent=silent)
    GRAPH_kernels = Graph_Hawkes(estimator_kernel, the_update_functions)
    GRAPH_kernels.estimation_hawkes_parameter_over_time(separator_colour='weight function')
    estimator_kernel.DF.to_csv(r'/home/bianca/est/estimators.csv', index=False,
                               header=True)

# I might have changed the parameters here in the code, so I come back to original version.
HAWKSY = Hawkes_process(tt, PARAMETERS)



#-----------------------------------------------------------------------------------------------
do = True ###################################### MSE
if do:
    estimator_MSE = Estimator_Hawkes()
    TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T, 40 * mini_T, 45 * mini_T,
             50 * mini_T, 60 * mini_T, 75 * mini_T, 90 * mini_T, 100 * mini_T, 110 * mini_T, 120 * mini_T, 130 * mini_T,
             140 * mini_T, 150 * mini_T]
    TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T]
    # , 20 * mini_T, 25 * mini_T, 30 * mini_T]
    count_times = 0
    for times in TIMES:
        count_times += 1
        print("=" * 78)
        print(
            "Time : {} out of : {}.".format(count_times, len(TIMES)))
        functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE,
                                                    times, nb_of_guesses, silent=silent)
    GRAPH_MSE = Graph_Hawkes(estimator_MSE, the_update_functions)
    GRAPH_MSE.convergence_estimators_limit_time(mini_T, TIMES, 'T_max', recurrent_functions.compute_MSE)








#-----------------------------------------------------------------------------------------------
do = False  ######################################
if do :
    functions_change_point_analysis.change_point_plot(r'/home/bianca/est/estimators_new.csv',
                      width = 5, min_size = 5, n_bkps=1, model="l2", column_for_multi_plot_name= 'weight function')








#-----------------------------------------------------------------------------------------------
do = True ######################################

#-----------------------------------------------------------------------------------------------
do = True ######################################

#-----------------------------------------------------------------------------------------------
do = True ######################################

#-----------------------------------------------------------------------------------------------
do = True ######################################






# print("------------------------------------------------------------------------------------")
# print("------------------------------- estimation -----------------------------------------")
# print("------------------------------------------------------------------------------------")
# FIXME one_long_and_longer_estimation(tt, ALPHA, BETA, MU, mini_T)


#
# kernel = Kernel(fct_top_hat, name = "wide top hat")
# w = kernel.eval(T_t = time_real, eval_point = 0)









plt.show()