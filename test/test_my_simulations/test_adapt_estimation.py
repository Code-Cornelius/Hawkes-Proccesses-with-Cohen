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
h = 2.5
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
            print(
                f"Time : {count_times} out of : {len(Times)}. Kernel : {count_kernels} out of : {len(list_of_kernels)}.")
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

    def test_over_the_time_simple_draw(self):
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\5-kernels-over_the_time.csv'

        if test_mode:
            list_of_kernels = [Kernel(fct_truncnorm, name="large truncnorm", a=-b, b=b, sigma=b * 0.6)]
        else:
            list_of_kernels = [  # Kernel(fct_truncnorm, name="my truncnorm", a=-350, b=350, sigma=300),
                Kernel(fct_truncnorm, name="large truncnorm", a=-b, b=b, sigma=b * 0.6),
                # Kernel(fct_truncnorm, name="large, high truncnorm", a= -b, b = b, sigma= b * 0.9),
                Kernel(fct_top_hat, name="top-hat", a=-b, b=b),
                Kernel(fct_biweight, name="biweight", a=-b, b=b),
                Kernel(fct_epa, name="epanechnikov", a=-b, b=b)
            ]
            # list_of_kernels = [  Kernel(fct_biweight, name="biweight small", a=-b/2, b=b/2),
            #                      Kernel(fct_biweight, name="biweight medium", a=-b, b=b),
            #                      Kernel(fct_biweight, name="biweight large", a=-b*1.5, b=b*1.5),
            #                   ]

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)

        plot_param = list_of_kernels, Times[NB_OF_TIMES // 2]  # we plot the kernel in the middle

        evol_graph = Evolution_plot_estimator_Hawkes.from_path(path, the_update_functions)
        # the parameter I am giving says to plot only one kernel on the graph.
        evol_graph.draw(separator_colour='weight function',
                        one_kernel_plot_param=plot_param)

    def test_over_the_time_adaptive_one_simulate(self):
        path = first_estimation_path

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

        estimator_kernel.to_csv(path,
                                index=False,
                                header=True)

        evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
        list_of_kernels = []
        for i in range(len(Times)):
            list_of_kernels.append(my_opt_kernel)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function', kernel_plot_param=plot_param)

    def test_over_the_time_adaptive_one_draw(self):
        #  put optimal kernel here
        my_opt_kernel = Kernel(fct_biweight, name=f"Biweight {width_kernel} width", a=-b, b=b)
        #todo ce que je peux faire c'est lire les temps depuis le DF, pour ça,  my_df['time estimation'].unique()

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)

        evol_graph = Evolution_plot_estimator_Hawkes.from_path(first_estimation_path, the_update_functions)
        list_of_kernels = []
        for i in range(len(Times)):
            list_of_kernels.append(my_opt_kernel)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function',
                        kernel_plot_param=plot_param)

    def test_over_the_time_adaptive_two_simulate(self):
        #todo ce que je peux faire c'est lire les temps depuis le DF, pour ça,  my_df['time estimation'].unique()

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)
        estimator_kernel = Estimator_Hawkes.from_path(first_estimation_path)

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
        evol_kernels.draw(separator_colour='weight function', kernel_plot_param=plot_param)
        adaptive_estimator_kernel.to_csv(second_estimation_path, index=False, header=True)

    def test_over_the_time_adaptive_two_draw(self):
        path_for_first_simul = first_estimation_path
        path_for_second_simul = second_estimation_path

        #todo ce que je peux faire c'est lire les temps depuis le DF, pour ça,  my_df['time estimation'].unique()

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)
        estimator_kernel = Estimator_Hawkes.from_path(path_for_first_simul)

        _, list_of_kernels = functions_fct_rescale_adaptive.creator_kernels_adaptive(
            my_estimator_mean_dict=estimator_kernel, Times=Times, considered_param=CONSIDERED_PARAM,
            list_previous_half_width=[b] * NB_OF_TIMES, L=L, R=R, h=h, l=l, tol=0.1, silent=silent)

        evol_graph = Evolution_plot_estimator_Hawkes.from_path(path_for_first_simul, the_update_functions)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function',
                        kernel_plot_param=plot_param)

    def test_comparison_before_after_rescale(self):
        path_for_first_simul = first_estimation_path
        path_for_second_simul = second_estimation_path

        df_1 = pd.read_csv(path_for_first_simul)
        df_2 = pd.read_csv(path_for_second_simul)
        my_df = pd.concat([df_1, df_2], axis=0, join='outer', ignore_index=False, keys=None,
                          levels=None, names=None, verify_integrity=False, copy=True)
        #todo ce que je peux faire c'est lire les temps depuis le DF, pour ça,  my_df['time estimation'].unique()

        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)
        estimator_kernel = Estimator_Hawkes(df_1)

        _, list_of_kernels = functions_fct_rescale_adaptive.creator_kernels_adaptive(
            my_estimator_mean_dict=estimator_kernel, Times=Times, considered_param=CONSIDERED_PARAM,
            list_previous_half_width=[b] * NB_OF_TIMES, L=L, R=R, h=h, l=l, tol=0.1, silent=silent)
        my_estim = Estimator_Hawkes(my_df)
        evol_graph = Evolution_plot_estimator_Hawkes(my_estim, the_update_functions)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function',
                        kernel_plot_param=plot_param,
                        all_kernels_drawn=False)

    def test_comparison_three_estimations_draw(self):
        path_for_first_simul  = first_estimation_path
        path_for_second_simul = second_estimation_path
        path_for_third_simul  = third_estimation_path

        df_1 = pd.read_csv(path_for_first_simul)
        df_2 = pd.read_csv(path_for_second_simul)
        df_3 = pd.read_csv(path_for_third_simul)

        my_total_df = pd.concat([df_1, df_2, df_3], axis=0, join='outer', ignore_index=False, keys=None,
                          levels=None, names=None, verify_integrity=False, copy=True)
        #todo ce que je peux faire c'est lire les temps depuis le DF, pour ça,  my_df['time estimation'].unique()
        Times = np.linspace(Test_Simulation_Hawkes_adaptive.lower_percent_bound * T_max,
                            Test_Simulation_Hawkes_adaptive.higher_percent_bound * T_max, NB_OF_TIMES)
        estimator_kernel_total = Estimator_Hawkes(my_total_df)
        estimator_kernel_df1 = Estimator_Hawkes(df_1)
        estimator_kernel_df2 = Estimator_Hawkes(df_2)

        #first list of kernels.
        list_half_width, _ = functions_fct_rescale_adaptive.creator_kernels_adaptive(
            my_estimator_mean_dict=estimator_kernel_df1, Times=Times, considered_param=CONSIDERED_PARAM,
            list_previous_half_width=[b] * NB_OF_TIMES, L=L, R=R, h=h, l=l, tol=0.1, silent=silent)

        # we take back the kernels for the second plot.
        _, list_of_kernels = functions_fct_rescale_adaptive.creator_kernels_adaptive(
            my_estimator_mean_dict=estimator_kernel_df2, Times=Times, considered_param=CONSIDERED_PARAM,
            list_previous_half_width=list_half_width, L=L, R=R, h=h, l=l, tol=0.1, silent=silent)
        evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel_total, the_update_functions)
        plot_param = list_of_kernels, Times
        # I am plotting many kernels here.
        evol_graph.draw(separator_colour='weight function',
                        kernel_plot_param=plot_param,
                        all_kernels_drawn=True)
