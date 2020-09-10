from test.test_my_simulations.setup_for_estimations import *

##### normal libraries
import unittest

##### my libraries
from library_functions.tools import decorators_functions
from functions import functions_fct_rescale_adaptive

##### other files
from functions import functions_for_MLE
from classes.graphs.class_graph_estimator_hawkes import *
from classes.graphs.class_evolution_plot_estimator_Hawkes import Evolution_plot_estimator_Hawkes
from classes.class_hawkes_process import *

L = L_PARAM
R = R_PARAM
h = h_PARAM
if l_PARAM == "automatic with respect to the total size":
    l = width_kernel / T_max / 2


class Test_Simulation_Hawkes_adaptive(unittest.TestCase):
    # section ######################################################################
    #  #############################################################################
    # setup

    higher_percent_bound = 0.95
    lower_percent_bound = 0.05

    def setUp(self):
        # check:
        print("L : {}, R : {}, T_max : {}, width : {}".format(L, R, T_max / 120, width_kernel / T_max))
        pass

    def tearDown(self):
        plt.show()

    # section ######################################################################
    #  #############################################################################
    # tests

    def test_over_the_time_simple_simulate(self):
        print("width of the kernels: {}.".format(width_kernel))

        estimator_kernel = Estimator_Hawkes()

        list_of_kernels = [Kernel(fct_biweight, name=f"Biweight {width_kernel} width", a=-b, b=b)]

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)

        total_nb_tries = len(Times) * len(list_of_kernels)
        actual_state = [0]  # initialization

        @decorators_functions.prediction_total_time(total_nb_tries=total_nb_tries,
                                                    multiplicator_factor=0.9,
                                                    actual_state=actual_state)
        def simulation(count_times, Times, count_kernels, list_of_kernels, HAWKSY, estimator_kernel, tt, nb_of_guesses,
                       kernel, a_time, silent):
            print(''.join(["\n", "=" * 78]))
            print(f"Time : {count_times} out of : {len(Times)}. Kernel : {count_kernels} out of : {len(list_of_kernels)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, tt=tt,
                                                            nb_of_guesses=nb_of_guesses,
                                                            kernel_weight=kernel, time_estimation=a_time, silent=silent)

        count_times = 0
        for a_time in Times:
            count_kernels = 0
            count_times += 1
            print(HAWKSY(a_time, T_max))
            for kernel in list_of_kernels:
                count_kernels += 1
                actual_state[0] += 1
                simulation(count_times, Times, count_kernels, list_of_kernels, HAWKSY, estimator_kernel, tt,
                           nb_of_guesses, kernel, a_time, silent)
        plot_param = list_of_kernels, Times[NB_OF_TIMES // 2]  # we plot the kernel in the middle
        evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
        evol_graph.draw(separator_colour='weight function', one_kernel_plot_param=plot_param)
        estimator_kernel.DF.to_csv(trash_path,
                                   index=False,
                                   header=True)


    def test_over_the_time_adaptive_one_simulate(self):
        path_for_saving = first_estimation_path

        estimator_kernel = Estimator_Hawkes()
        #  put optimal kernel here
        my_opt_kernel = Kernel(fct_biweight, name=f"Biweight {width_kernel} width", a=-b, b=b)
        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)
        actual_state = [0]  # initialization

        @decorators_functions.prediction_total_time(total_nb_tries=len(Times),
                                                    multiplicator_factor=0.9,
                                                    actual_state=actual_state)
        def simulation(count_times, Times, HAWKSY, estimator_kernel, tt, nb_of_guesses, my_opt_kernel, a_time, silent):
            print(''.join(["\n", "=" * 78]))
            print(f"Time : {count_times} out of : {len(Times)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, tt=tt,
                                                            nb_of_guesses=nb_of_guesses,
                                                            kernel_weight=my_opt_kernel, time_estimation=a_time,
                                                            silent=silent)

        ############################## first step
        count_times = 0
        for a_time in Times:
            print(HAWKSY(a_time, T_max))
            count_times += 1
            actual_state[0] += 1
            simulation(count_times, Times, HAWKSY, estimator_kernel, tt, nb_of_guesses, my_opt_kernel, a_time=a_time,
                       silent=silent)

        estimator_kernel.to_csv(path_for_saving,
                                index=False,
                                header=True)

        evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
        list_of_kernels = []
        for i in range(len(Times)):
            list_of_kernels.append(my_opt_kernel)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function', kernel_plot_param=plot_param)

    def test_over_the_time_adaptive_two_simulate(self):
        path_for_first_simul = first_estimation_path
        path_for_second_simul = second_estimation_path

        Times = pd.read_csv(path_for_first_simul)['time estimation'].unique()
        estimator_kernel = Estimator_Hawkes.from_path(path_for_first_simul)

        _, list_of_kernels = functions_fct_rescale_adaptive.creator_kernels_adaptive(
            my_estimator_mean_dict=estimator_kernel, Times=Times, considered_param=CONSIDERED_PARAM,
            list_previous_half_width=[b] * NB_OF_TIMES, L=l, R=R, h=h, l=l, tol=0.1, silent=silent)

        adaptive_estimator_kernel = Estimator_Hawkes()
        actual_state = [0]  # initialization

        @decorators_functions.prediction_total_time(total_nb_tries=len(Times),
                                                    multiplicator_factor=0.9,
                                                    actual_state=actual_state)
        def simulation(count_times, Times, HAWKSY, adaptive_estimator_kernel, tt, nb_of_guesses, kernel, a_time,
                       silent):
            print(''.join(["\n", "=" * 78]))
            print(f"Time : {count_times} out of : {len(Times)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, adaptive_estimator_kernel, tt=tt,
                                                            nb_of_guesses=nb_of_guesses,
                                                            kernel_weight=kernel, time_estimation=a_time,
                                                            silent=silent)

        ############################## second step
        count_times = 0

        for a_time, kernel in zip(Times, list_of_kernels):
            print(HAWKSY(a_time, T_max))
            count_times += 1
            actual_state[0] += 1
            simulation(count_times, Times, HAWKSY, adaptive_estimator_kernel, tt, nb_of_guesses, kernel, a_time, silent)

        evol_kernels = Evolution_plot_estimator_Hawkes(adaptive_estimator_kernel, the_update_functions)
        plot_param = list_of_kernels, Times
        evol_kernels.draw(separator_colour='weight function', kernel_plot_param=plot_param,
                        all_kernels_drawn=ALL_KERNELS_DRAWN)
        adaptive_estimator_kernel.to_csv(path_for_second_simul, index=False, header=True)
