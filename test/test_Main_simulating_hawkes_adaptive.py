import decorators_functions
import functions_change_point_analysis
import functions_fct_rescale_adaptive
from test.test_Main_simulating_hawkes_simple import *

class Test_Simulation_Hawkes_adaptive(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        plt.show()

    def test_over_the_time_simple(self):
        to_be_simulated = False
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\5-kernels-over_the_time.csv'

        if test_mode:
            nb_of_times = 5
        else :
            nb_of_times = 50

        width_kernel = 1 / 5. * T_max
        b = width_kernel / 2.
        print("width of the kernels: {}.".format(width_kernel))

        estimator_kernel = Estimator_Hawkes()

        if test_mode :
            list_of_kernels = [Kernel(fct_truncnorm, name="large truncnorm", a=-b, b=b, sigma=b * 0.6)]
        else :
            list_of_kernels = [  # Kernel(fct_truncnorm, name="my truncnorm", a=-350, b=350, sigma=300),
                Kernel(fct_truncnorm, name="large truncnorm", a= -b, b= b, sigma= b * 0.6),
                #Kernel(fct_truncnorm, name="large, high truncnorm", a= -b, b = b, sigma= b * 0.9),
                Kernel(fct_top_hat, name="top-hat", a=-b, b=b),
                Kernel(fct_biweight, name="biweight", a=-b, b=b),
                Kernel(fct_epa, name="epanechnikov", a=-b, b=b)
                            ]
            # list_of_kernels = [  Kernel(fct_biweight, name="biweight small", a=-b/2, b=b/2),
            #                      Kernel(fct_biweight, name="biweight medium", a=-b, b=b),
            #                      Kernel(fct_biweight, name="biweight large", a=-b*1.5, b=b*1.5),
            #                   ]

        Times = np.linspace(0.05 * T_max, 0.95 * T_max, nb_of_times)

        total_nb_tries = len(Times) * len(list_of_kernels)
        actual_state = [0]  # initialization

        if to_be_simulated:
            @decorators_functions.prediction_total_time(total_nb_tries=total_nb_tries,
                                                        multiplicator_factor=0.9,
                                                        actual_state=actual_state)
            def simulation(count_times, Times, count_kernels, list_of_kernels, HAWKSY, estimator_kernel, tt, nb_of_guesses, kernel, a_time, silent):
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
            plot_param = list_of_kernels, Times[nb_of_times//2] # we plot the kernel in the middle
            evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
            evol_graph.draw(separator_colour='weight function', one_kernel_plot_param= plot_param)
            estimator_kernel.DF.to_csv(trash_path,
                                       index=False,
                                       header=True)
        else :
            plot_param = list_of_kernels, Times[nb_of_times//2] # we plot the kernel in the middle

            evol_graph = Evolution_plot_estimator_Hawkes.from_path(path, the_update_functions)
            # the parameter I am giving says to plot only one kernel on the graph.
            evol_graph.draw(separator_colour='weight function',
                               one_kernel_plot_param= plot_param)



    def test_over_the_time_adaptive_one(self):
        to_be_simulated = True
        path = first_estimation_path
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\super_4_first.csv'


        if test_mode:
            nb_of_times = 3
        else:
            nb_of_times = 50

        width_kernel = 1 / 5 * T_max
        b = width_kernel / 2

        estimator_kernel = Estimator_Hawkes()
        #  put optimal kernel here
        my_opt_kernel = Kernel(fct_biweight, name=f"Biweight {width_kernel} width", a=-b, b=b)
        Times = np.linspace(0.05 * T_max, 0.95 * T_max, nb_of_times)
        actual_state = [0]  # initialization

        if to_be_simulated:
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
                simulation(count_times, Times, HAWKSY, estimator_kernel, tt, nb_of_guesses, my_opt_kernel, a_time, silent)

            estimator_kernel.to_csv(first_estimation_path,
                                    index=False,
                                    header=True)

            evol_graph = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
            list_of_kernels = []
            for i in range(len(Times)):
                list_of_kernels.append(my_opt_kernel)
            plot_param = list_of_kernels, Times
            # I am plotting many kernels here.
            evol_graph.draw(separator_colour='weight function', kernel_plot_param=plot_param)

        else :
            evol_graph = Evolution_plot_estimator_Hawkes.from_path(path, the_update_functions)
            list_of_kernels = []
            for i in range(len(Times)):
                list_of_kernels.append(my_opt_kernel)
            plot_param = list_of_kernels, Times
            # I am plotting many kernels here.
            evol_graph.draw(separator_colour='weight function',
                               kernel_plot_param=plot_param)

    def test_over_the_time_adaptive_two(self):
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\super_4_first.csv'

        considered_param = ['nu','alpha','beta']

        if test_mode:
            nb_of_times = 3
        else:
            nb_of_times = 50

        width_kernel = 1 / 5. * T_max
        b = width_kernel / 2.
        Times = np.linspace(0.05 * T_max, 0.95 * T_max, nb_of_times)
        estimator_kernel = Estimator_Hawkes.from_path(path)

        list_of_kernels = functions_fct_rescale_adaptive.creator_kernels_adaptive(my_estimator_mean_dict = estimator_kernel, Times = Times,
                                 considered_param = considered_param, half_width = b, tol = 0.1, silent=silent)

        adaptive_estimator_kernel = Estimator_Hawkes()
        actual_state = [0]  # initialization

        @decorators_functions.prediction_total_time(total_nb_tries=len(Times),
                                                    multiplicator_factor=0.9,
                                                    actual_state=actual_state)
        def simulation(count_times, Times, HAWKSY, adaptive_estimator_kernel, tt, nb_of_guesses, kernel, a_time, silent):
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











    def test_change_point_analysis(self):
        functions_change_point_analysis.change_point_plot(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            width=5, min_size=5, n_bkps=1, model="l2", column_for_multi_plot_name='weight function')