##### normal libraries


##### my libraries

##### other files
import unittest

from classes.class_Graph_Estimator_Hawkes import *
from classes.class_hawkes_process import *
from classes.class_kernel import *
import functions_change_point_analysis
import functions_MLE
import functions_fct_evol_parameters
import functions_general_for_Hawkes

np.random.seed(124)


def update_functions(case):
    the_update_functions = functions_general_for_Hawkes.multi_list_generator(HAWKSY.M)
    if case == 1:
        for i in range(HAWKSY.M):
            the_update_functions[0][i] = \
                lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 0.1, MU[i], T_max)
            for j in range(HAWKSY.M):
                the_update_functions[1][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 2, ALPHA[i, j], T_max)
                the_update_functions[2][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.linear_growth(time, 3, BETA[i, j], T_max)

    elif case == 2:
        for i in range(HAWKSY.M):
            the_update_functions[0][i] = \
                lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.1, MU[i], 0, T_max)
            for j in range(HAWKSY.M):
                the_update_functions[1][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.7, ALPHA[i, j], ALPHA[i, j],
                                                                               T_max)
                the_update_functions[2][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.one_jump(time, 0.4, BETA[i, j], BETA[i, j], T_max)

    elif case == 3:
        for i in range(HAWKSY.M):
            the_update_functions[0][i] = \
                lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump=0.7, a=0, b=MU[i],
                                                                               base_value=MU[i] * 1.5, T_max=T_max)
            for j in range(HAWKSY.M):
                the_update_functions[1][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump=0.4, a=1.4,
                                                                                   b=ALPHA[i, j],
                                                                                   base_value=ALPHA[i, j] / 2,
                                                                                   T_max=T_max)
                the_update_functions[2][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.moutain_jump(time, when_jump=0.7, a=1.8,
                                                                                   b=BETA[i, j],
                                                                                   base_value=BETA[i, j] / 1.5,
                                                                                   T_max=T_max)

    elif case == 4:
        for i in range(HAWKSY.M):
            the_update_functions[0][i] = \
                lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, MU[i], 0.2)
            for j in range(HAWKSY.M):
                the_update_functions[1][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, ALPHA[i, j], 1)
                the_update_functions[2][i][j] = \
                    lambda time, T_max: functions_fct_evol_parameters.periodic_stop(time, T_max, BETA[i, j], 2.5)
    return the_update_functions


def choice_parameter(dim, styl):
    # dim choses how many dimensions
    # styl choses which variant of the parameters.
    if dim == 1:
        ALPHA = [[1.75]]
        BETA = [[2]]
        MU = [0.2]
        T0, mini_T = 0, 35  # 50 jumps for my uni variate stuff

    if dim == 2:
        ALPHA = [[2, 1],
                 [1, 2]]
        BETA = [[5, 3],
                [3, 5]]
        MU = [0.2, 0.2]
        T0, mini_T = 0, 120

    ALPHA, BETA, MU = np.array(ALPHA, dtype=np.float), np.array(BETA, dtype=np.float), np.array(MU,
                                                                                                dtype=np.float)  # I precise the type because he might think the np.array is int type.
    PARAMETERS = [MU.copy(), ALPHA.copy(), BETA.copy()]

    print("ALPHA : \n", ALPHA)
    print("BETA : \n", BETA)
    print('MU : \n', MU)
    print("=" * 78)
    print("=" * 78)
    print("=" * 78)
    return PARAMETERS, ALPHA, BETA, MU, T0, mini_T


##########################################
################parameters################
##########################################
# 2000 JUMPS
# T = 200 * mini_T
####################################################################### TIME
# number of max jump
nb_of_sim, M_PREC = 50000, 200000
M_PREC += 1
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
# simulation
silent = True
test_mode = False
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
PARAMETERS, ALPHA, BETA, MU, T0, mini_T = choice_parameter(1, 0)
estimator_multi = Estimator_Hawkes()
if test_mode:
    nb_of_guesses, T = 3, 30 * mini_T
else:
    nb_of_guesses, T = 50, 120 * mini_T
# a good precision is 500*(T-T0)
tt = np.linspace(T0, T, M_PREC, endpoint=True)
HAWKSY = Hawkes_process(tt, PARAMETERS)
trash_path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators.csv'


class Test_Simulation_Hawkes(unittest.TestCase):

    def setUp(self):
        self.the_update_functions = update_functions(3)

    def tearDown(self):
        plt.show()

    def test_plot_hawkes(self):
        intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool=True, silent=True)
        HAWKSY.plot_hawkes(time_real, intensity, name="EXACT_HAWKES")

    def test_simple_unique(self):
        intensity, time_real = HAWKSY.simulation_Hawkes_exact(T_max=T, plot_bool=False, silent=silent)
        print(functions_MLE.call_newton_raph_MLE_opt(time_real, T, silent=silent))
        self.assertTrue(True)

    def test_from_csv(self):
        graph_test = Graph_Estimator_Hawkes.from_path(
            trash_path,
            self.the_update_functions)
        graph_test.draw_evolution_parameter_over_time(separator_colour='weight function')
        graph_test.draw_histogram()
        TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T]
        graph_test = Graph_Estimator_Hawkes.from_path(
            trash_path, self.the_update_functions)
        graph_test.convergence_estimators_limit(mini_T, TIMES, 'T_max', recurrent_functions.compute_MSE)

    def test_simple_multi(self):
        estimator_multi = Estimator_Hawkes()
        functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_multi, T, nb_of_guesses, silent=silent)
        GRAPH_multi = Graph_Estimator_Hawkes(estimator_multi, self.the_update_functions)
        GRAPH_multi.draw_histogram()

        estimator_multi.to_csv(trash_path, index=False, header=True)

    def test_over_the_time(self):
        nb_of_times = 10
        HAWKSY = Hawkes_process(tt, PARAMETERS)
        # I create here the array. It is quite hard because I want a list of size size*size*3 where all elements can be change however I want. Other ways lead dependant vectors.

        # Here I change the parameters over time.
        estimator_kernel = Estimator_Hawkes()
        list_of_kernels = [Kernel(fct_truncnorm, name="my truncnorm", a=-350, b=350, sigma=300),
                           Kernel(fct_truncnorm, name="large truncnorm", a=-500, b=500, sigma=300),
                           Kernel(fct_truncnorm, name="large, high truncnorm", a=-500, b=500, sigma=450)]
        Times = np.linspace(0.1 * T, 0.9 * T, nb_of_times)

        count_kernels = 0
        count_times = 0
        for time in Times:
            count_kernels = 0
            count_times += 1
            HAWKSY.update_coef(time, self.the_update_functions, T_max=T)
            print(HAWKSY)
            for kernel in list_of_kernels:
                count_kernels += 1
                print("=" * 78)
                print("Time : {} out of : {}. Kernel : {} out of : {}.".format(count_times, len(Times), count_kernels,
                                                                               len(list_of_kernels)))
                functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max=T,
                                                            nb_of_guesses=nb_of_guesses,
                                                            kernel_weight=kernel, time_estimation=time, silent=silent)
        GRAPH_kernels = Graph_Estimator_Hawkes(estimator_kernel, self.the_update_functions)
        GRAPH_kernels.draw_evolution_parameter_over_time(separator_colour='weight function')
        estimator_kernel.DF.to_csv(trash_path,
                                   index=False,
                                   header=True)

    def test_MSE(self):
        estimator_MSE = Estimator_Hawkes()

        TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T, 40 * mini_T, 45 * mini_T,
                 50 * mini_T, 60 * mini_T, 75 * mini_T, 90 * mini_T, 100 * mini_T, 110 * mini_T, 120 * mini_T,
                 130 * mini_T,
                 140 * mini_T, 150 * mini_T]
        TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T]
        count_times = 0
        for times in TIMES:
            count_times += 1
            print("=" * 78)
            print(
                "Time : {} out of : {}.".format(count_times, len(TIMES)))
            functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE,
                                                        times, nb_of_guesses, silent=silent)
        GRAPH_MSE = Graph_Estimator_Hawkes(estimator_MSE, self.the_update_functions)
        estimator_MSE.DF.to_csv(trash_path, index=False,
                                header=True)
        GRAPH_MSE.convergence_estimators_limit(mini_T, TIMES, 'T_max', recurrent_functions.compute_MSE)

    def test_change_point_analysis(self):
        functions_change_point_analysis.change_point_plot(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            width=5, min_size=5, n_bkps=1, model="l2", column_for_multi_plot_name='weight function')


plt.show()