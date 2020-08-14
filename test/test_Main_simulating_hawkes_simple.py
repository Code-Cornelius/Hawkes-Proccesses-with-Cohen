##### normal libraries
import unittest

##### my libraries

##### other files
from classes.class_Graph_Estimator_Hawkes import *
from classes.class_hawkes_process import *
import functions_for_MLE
from functions_fct_evol_parameters import update_functions, constant_parameter, linear_growth, one_jump, moutain_jump, \
    periodic_stop

np.random.seed(124)


def choice_parameter(dim, styl):
    # dim choses how many dimensions
    # styl choses which variant of the parameters.
    if dim == 1:
        if styl ==1:
            ALPHA = [[1.2]]
            BETA = [[2]]
            MU = [0.2]
            T0, mini_T = 0, 120  # 50 jumps for my uni variate stuff
        elif styl == 2:
            ALPHA = [[2.]]
            BETA = [[2.4]]
            MU = [0.2]
            T0, mini_T = 0, 45  # 50 jumps for my uni variate stuff
        elif styl == 3:
            ALPHA = [[1.75]]
            BETA = [[2]]
            MU = [0.5]
            T0, mini_T = 0, 15  # 50 jumps for my uni variate stuff
        elif styl ==4:
            ALPHA = [[1]]
            BETA = [[4]]
            MU = [0.2]
            T0, mini_T = 0, 45  # 50 jumps for my uni variate stuff


    if dim == 2:
        if styl ==1:
            ALPHA = [[2, 1],
                     [1, 2]]
            BETA = [[7, 4],
                    [4, 7]]
            MU = [0.2, 0.2]
            T0, mini_T = 0, 70
        if styl == 2:
            ALPHA = [[2, 2],
                     [1, 2]]
            BETA = [[5, 3],
                    [3, 5]]
            MU = [0.4, 0.3]
            T0, mini_T = 0, 12

    if dim == 5:
        ALPHA = [[2, 1, 0.5, 0.5, 0.5],
                 [1, 2, 0.5, 0.5, 0.5],
                 [0, 0, 0.5, 0, 0],
                 [0, 0, 0., 0.5, 0.5],
                 [0, 0, 0., 0.5, 0.5]]
        BETA = [[5, 5, 5, 6, 3],
                [5, 5, 5, 6, 3],
                [0, 0, 10, 0, 0],
                [0, 0, 0, 6, 3],
                [0, 0, 0, 6, 3]]
        MU = [0.2, 0.2, 0.2, 0.2, 0.2]
        T0, mini_T = 0, 5

    ALPHA, BETA, MU = np.array(ALPHA, dtype=np.float), np.array(BETA, dtype=np.float), np.array(MU,
                                                                                                dtype=np.float)  # I precise the type because he might think the np.array is int type.
    PARAMETERS = [MU.copy(), ALPHA.copy(), BETA.copy()]

    print("ALPHA : \n", ALPHA)
    print("BETA : \n", BETA)
    print('NU : \n', MU)
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

# section ######################################################################
#  #############################################################################
# simulation

silent = True
test_mode = False

# section ######################################################################
#  #############################################################################
print("\n~~~~~Computations.~~~~~\n")
dim = 1
PARAMETERS, ALPHA, BETA, MU, T0, mini_T = choice_parameter(dim = dim  , styl = 1)
print(PARAMETERS)
the_update_functions = update_functions(2, PARAMETERS)


estimator_multi = Estimator_Hawkes()

if test_mode:
    nb_of_guesses, T = 3, 50 * mini_T
else:
    nb_of_guesses, T = 50,  80 * mini_T #in terms of how many jumps, I want roughly 7500 jumps
# a good precision is 500*(T-T0)
tt = np.linspace(T0, T, M_PREC, endpoint=True)



HAWKSY = Hawkes_process(the_update_functions)
# for not keeping the data, I store it in the bin:
trash_path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators.csv'
# for the first estimate in the adaptive streategy I store it there:
first_estimation_path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators_first.csv'
second_estimation_path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators_second.csv'


class Test_Simulation_Hawkes_simple(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        plt.show()

    def test_form_evol_functions(self):
        for i in range(5):
            the_update_functions = update_functions(i, PARAMETERS)
            Hawkes_process(the_update_functions)

    def test_plot_hawkes(self):
        intensity, time_real = HAWKSY.simulation_Hawkes_exact_with_burn_in(tt= tt, plot_bool=True, silent=False)
        HAWKSY.plot_hawkes(tt, time_real, intensity, name="EXACT_HAWKES")

    def test_simple_unique(self):
        _, time_real = HAWKSY.simulation_Hawkes_exact_with_burn_in(tt = tt, plot_bool=False, silent=True)
        print(len(time_real[0]))
        print(functions_for_MLE.call_newton_raph_MLE_opt(time_real, T, silent=False))

    def test_from_csv(self):
        graph_test = Graph_Estimator_Hawkes.from_path(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            the_update_functions)
        graph_test.draw_evolution_parameter_over_time(separator_colour='weight function')
        graph_test.draw_histogram()
        TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T]
        graph_test = Graph_Estimator_Hawkes.from_path(
            'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators_test.csv',
            the_update_functions)
        graph_test.convergence_estimators_limit(mini_T, TIMES, 'T_max', recurrent_functions.compute_MSE)

    def test_simple_multi(self):
        estimator_multi = Estimator_Hawkes()
        functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_multi, tt, nb_of_guesses, silent=silent)
        GRAPH_multi = Graph_Estimator_Hawkes(estimator_multi, the_update_functions)
        GRAPH_multi.draw_histogram()

        estimator_multi.to_csv(trash_path, index=False, header=True)

    def test_MSE(self):
        estimator_MSE = Estimator_Hawkes()

        TIMES = [10 * mini_T, 20 * mini_T, 30 * mini_T, 40 * mini_T, 45 * mini_T,
                 50 * mini_T, 60 * mini_T, 75 * mini_T, 90 * mini_T, 100 * mini_T, 120 * mini_T,
                 130 * mini_T, 140 * mini_T, 150 * mini_T, 160 * mini_T]
        # TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T]
        count_times = 0
        for times in TIMES:
            tt = np.linspace(T0, times, M_PREC, endpoint=True)
            count_times += 1
            print("=" * 78)
            print(
                f"Time : {count_times} out of : {len(TIMES)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE,
                                                            tt, nb_of_guesses, silent=silent)
        GRAPH_MSE = Graph_Estimator_Hawkes(estimator_MSE, the_update_functions)
        estimator_MSE.DF.to_csv(trash_path, index=False,
                                header=True)
        GRAPH_MSE.convergence_estimators_limit(mini_T, TIMES, 'T_max', recurrent_functions.compute_MSE)

    def test_capabilities_test_optimization(self):
        estimator_MSE = Estimator_Hawkes()
        T = [10 * mini_T, 20 * mini_T, 30 * mini_T, 40 * mini_T,
             50 * mini_T, 60 * mini_T, 70 * mini_T, 80 * mini_T, 90 * mini_T, 100 * mini_T,
             120 * mini_T, 150 * mini_T, 180 * mini_T, 200 * mini_T, 240 * mini_T]
        T_plot = [T[i] // mini_T * 50 for i in range(len(T))]
        vect_time_simulation = np.zeros(len(T))

        for i, times in enumerate(T):
            tt = np.linspace(T0, times, M_PREC, endpoint=True)
            print(f"times {times}")
            start = time.time()
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE, tt=tt,
                                                            nb_of_guesses=nb_of_guesses,
                                                            silent=silent)
            time_simulation = time.time() - start
            vect_time_simulation[i] = time_simulation / nb_of_guesses

        aplot = APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=T_plot, yy=vect_time_simulation)
        aplot.set_dict_fig(0, {
            'title': "Increase in time for simulation and convergence of the estimation for Hawkes processes, batches of {} realisations.".format(
                nb_of_guesses),
            'xlabel': "Number of Jumps simulated", 'ylabel': "Average time to simulate"})
        aplot.save_plot("Timing_opti")
        return