import numpy as np
import statistics as stat


import functions_MLE
import scipy.stats

from generic_functions import *
import plot_functions
from useful_functions import *

from operator import itemgetter  # at some point I need to get the list of ranks of a list.

import class_kernel

from class_hawkes_process import *
from class_estimator import *
from class_graph import *
import functions_general_for_Hawkes
import functions_change_point_analysis

# a is already divided by t_max, so just put of how much you want to grow
def linear_growth(time, a, b, T_max):  # ax + b
    return a / T_max * time + b


# when jump should be a %
def one_jump(time, when_jump, original_value, new_value, T_max):
    return original_value + new_value * np.heaviside(time - T_max * when_jump, 1)


# when jump should be a %
def moutain_jump(time, when_jump, a, b, base_value, T_max):
    if time < when_jump * T_max:
        return linear_growth(time, a, b, T_max)
    else:
        return base_value


#
def periodic_stop(time, T_max, a, base_value): # need longer realisation like 80 mini_T
    if time / T_max * 2 * cmath.pi * 2.25 < 2 * cmath.pi * 1.75:
        return base_value + a*cmath.cos(time / T_max * 2 * cmath.pi * 2.25) * cmath.cos(time / T_max * 2 * cmath.pi * 2.25)
    else:
        return base_value


##########################################
################parameters################
##########################################

# timing
T0, mini_T = 0, 35 # 50 jumps for my uni variate stuff

# so here I should have around 500 jumps.
#T = 10 * mini_T
# 2000 JUMPS
T = 200 * mini_T
T = 5 * mini_T


####################################################################### TIME
# number of max jump
nb_of_sim, M_PREC = 50000,100000
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
'''
ALPHA = [[2, 0], [0, 3]]
BETA = [[20, 20], [10, 10]]
MU = [0.7, 0.3]

#'''
#'''
ALPHA = [[1.75]]
BETA = [[2]]
MU = [0.2]
# '''
'''
ALPHA = [[0.6]]
BETA = [[1.2]]
MU = [1.2]
#'''
ALPHA, BETA, MU = np.array(ALPHA, dtype=np.float), np.array(BETA, dtype=np.float), np.array(MU, dtype=np.float) # I precise the type because he might think the np.array is int type.
print("ALPHA : \n ", ALPHA)
print("BETA : \n ", BETA)
print('MU : \n ', MU)
print("=" * 78)
print("=" * 78)
print("=" * 78)
np.random.seed(124)

HAWKSY = Hawkes_process(tt, ALPHA, BETA, MU)
estimator = pd.DataFrame(columns=['variable', 'n', 'm',
                                  'time estimation', 'weight function',
                                  'value', 'T_max', 'true value'])

graph = Graph.from_path(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_sin_long.csv',
                    50, [[[0]], [[0]], [0]])

print(graph)
time.sleep(1000)
estimator_multi = Estimator(estimator)
################################################
# plot
plot = False
################################################
if plot:
    intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool = True, silent = True)
    HAWKSY.plot_hawkes(time_real, intensity, name = "EXACT_HAWKES")
    plt.show()















################################################
# simulation
silent = True
test_mode = True
################################################
################################################
if test_mode :
    nb_of_guesses, T = 10, 40 * mini_T
else:
    nb_of_guesses, T = 50, 120 * mini_T
tt = np.linspace(T0, T, M_PREC, endpoint=True)
################################################
################################################
do = False ###################################### SIMPLE UNIQUE
if do:
    intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool = False, silent = silent)
    print( functions_MLE.call_newton_raph_MLE_opt(time_real, T, silent = silent) )









#-----------------------------------------------------------------------------------------------
do = True ###################################### SIMPLE MULTI
if do:
    estimator_multi = Estimator(estimator)
    functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_multi, T, nb_of_guesses, silent = silent)
    GRAPH_multi = Graph(estimator_multi, ALPHA, BETA, MU, T, nb_of_guesses)
    GRAPH_multi.multi_simul_Hawkes_and_estimation()
    estimator_multi.DF.to_csv(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators.csv', index = False, header=True)



#-----------------------------------------------------------------------------------------------
do = False ###################################### OVER THE TIME ESTIMATION, DIFFERENT KERNELS
nb_of_times = 45
if do:
    HAWKSY = Hawkes_process(tt, ALPHA, BETA, MU)
    #I create here the array. It is quite hard because I want a list of size size*size*3 where all elements can be change however I want. Other ways lead dependant vectors.
    the_update_functions = functions_general_for_Hawkes.multi_list_generator(HAWKSY.M)
    
    # choice of case study
    #########
    case = 4
    #########
    if case == 1:
        the_update_functions[0][0][0] = lambda time, T_max: linear_growth(time, 2, ALPHA[0, 0], T_max)
        the_update_functions[1][0][0] = lambda time, T_max: linear_growth(time, 3, BETA[0, 0], T_max)
        the_update_functions[2][0][0] = lambda time, T_max: linear_growth(time, 0.1, MU[0], T_max)
    elif case == 2:
        the_update_functions[0][0][0] = lambda time, T_max: one_jump(time, 0.7, ALPHA[0, 0], ALPHA[0, 0], T_max)
        the_update_functions[1][0][0] = lambda time, T_max: one_jump(time, 0.4, BETA[0, 0], BETA[0, 0], T_max)
        the_update_functions[2][0][0] = lambda time, T_max: one_jump(time, 0.1, MU[0], 0, T_max)
    elif case == 3:
        the_update_functions[0][0][0] = lambda time, T_max: moutain_jump(time, 0.4, 1.4, ALPHA[0,0], ALPHA[0,0]/2, T_max)
        the_update_functions[1][0][0] = lambda time, T_max: moutain_jump(time, 0.7, 1.8, BETA[0,0], BETA[0,0]/1.5, T_max)
        the_update_functions[2][0][0] = lambda time, T_max: moutain_jump(time, 0.7, 0, MU[0], MU[0]*1.5, T_max)
    elif case == 4:
        the_update_functions[0][0][0] = lambda time, T_max: periodic_stop(time, T_max, ALPHA[0,0], 1)
        the_update_functions[1][0][0] = lambda time, T_max: periodic_stop(time, T_max, BETA[0,0], 2.5)
        the_update_functions[2][0][0] = lambda time, T_max: periodic_stop(time, T_max, MU[0], 0.2)




    # Here I change the parameters over time.
    estimator_kernel = Estimator(estimator)
    list_of_kernels = [ Kernel(fct_truncnorm, name = "high truncnorm", a = -350, b = 350, sigma = 300)]
    Times = np.linspace(0.1 * T, 0.9 * T, nb_of_times)

    count_kernels = 0 ; count_times = 0
    for time in Times:
        count_kernels = 0
        count_times += 1
        HAWKSY.update_coef(time, the_update_functions, T_max = T)
        print(HAWKSY)
        for kernel in list_of_kernels:
            count_kernels += 1
            print( "=" * 78)
            print( "Time : {} out of : {}. Kernel : {} out of : {}.".format(count_times, len(Times), count_kernels, len(list_of_kernels) ) )
            functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max = T, nb_of_guesses = nb_of_guesses,
                                                                           kernel_weight = kernel, time_estimation = time, silent=silent)
    GRAPH_kernels = Graph(estimator_kernel, ALPHA, BETA, MU, T, nb_of_guesses)
    GRAPH_kernels.estimation_hawkes_parameter_over_time(the_update_functions, T_max = T)
    estimator_kernel.DF.to_csv(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators.csv', index=False,
                              header=True)

    #functions_change_point_analysis.hange_point_plot('C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\RESEARCH COHEN\\estimators_kernel_sin_long.csv',
    #                  9, 5, column_for_multi_plot_name='weight function')
# I might have changed the parameters here in the code, so I come back to original version.
HAWKSY = Hawkes_process(tt, ALPHA, BETA, MU)








#-----------------------------------------------------------------------------------------------
do = False ###################################### MSE
if do :
    estimator_MSE = Estimator(estimator)
    TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T, 40 * mini_T, 45 * mini_T,
             50 * mini_T, 60 * mini_T, 75 * mini_T, 90 * mini_T, 100 * mini_T, 110 * mini_T, 120 * mini_T, 130 * mini_T,
             140 * mini_T, 150 * mini_T]
    TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T]
    count_times = 0
    for times in TIMES:
        count_times += 1
        print("=" * 78)
        print(
            "Time : {} out of : {}.".format(count_times, len(TIMES)))
        functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE,
                                                                      times, nb_of_guesses, silent = silent)
    GRAPH_MSE = Graph(estimator_MSE, ALPHA, BETA, MU, TIMES, nb_of_guesses)
    GRAPH_MSE.multi_simul_Hawkes_and_estimation_MSE(mini_T)





















#-----------------------------------------------------------------------------------------------
do = False ######################################
if do :
    estimator_CPA = Estimator(estimator)









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
#TODO one_long_and_longer_estimation(tt, ALPHA, BETA, MU, mini_T)


#
# kernel = Kernel(fct_top_hat, name = "wide top hat")
# w = kernel.eval(T_t = time_real, eval_point = 0)









plt.show()