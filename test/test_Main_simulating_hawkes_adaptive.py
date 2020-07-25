##### normal libraries
import unittest

from test.test_Main_simulating_hawkes_simple import *


class Test_Simulation_Hawkes_adaptive(unittest.TestCase):

    def setUp(self):
        self.the_update_functions = update_functions(1)

    def tearDown(self):
        plt.show()

    def test_over_the_time_simple(self):
        nb_of_times = 30
        # work-in-progress 25/07/2020 nie_k:  I will change the kernels for the fix width.
        width_kernel = 1/5
        b = width_kernel/2
        print( "width of the kernels: {}.".format(width_kernel))

        HAWKSY = Hawkes_process(tt, PARAMETERS)

        estimator_kernel = Estimator_Hawkes()
        list_of_kernels = [#Kernel(fct_truncnorm, name="my truncnorm", a=-350, b=350, sigma=300),
                           #Kernel(fct_truncnorm, name="large truncnorm", a=-500, b=500, sigma=300),
                           Kernel(fct_truncnorm, name="large, high truncnorm", a=-500, b=500, sigma=450),
                           #Kernel(fct_top_hat, name="top hat", a=-500, b=500),
                           Kernel(fct_truncnorm_test, name="test", a = -500, b = 500, sigma = 450)
                           #Kernel(fct_biweight, name="bi weight", a=-500, b=500),
                           #Kernel(fct_epa, name="epa", a=-500, b=500)
                           ]
        Times = np.linspace(0.05 * T, 0.95 * T, nb_of_times)
        Times = np.linspace(0.0 * T, 0.6 * T, nb_of_times)


        total_nb_tries = len(Times) * len(list_of_kernels)
        actual_state = [0] # initialization
        @decorators_functions.prediction_total_time(total_nb_tries=total_nb_tries,
                                                    multiplicator_factor=0.7,
                                                    actual_state=actual_state)
        def simulation():
            print(''.join(["\n", "=" * 78]))
            print(
                f"Time : {count_times} out of : {len(Times)}. Kernel : {count_kernels} out of : {len(list_of_kernels)}.")
            functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max=T,
                                                        nb_of_guesses=nb_of_guesses,
                                                        kernel_weight=kernel, time_estimation=time, silent=silent)

        count_times = 0
        for time in Times:
            count_kernels = 0
            count_times += 1
            HAWKSY.update_coef(time, self.the_update_functions, T_max=T)
            print(HAWKSY)
            for kernel in list_of_kernels:
                count_kernels += 1
                actual_state[0] += 1
                simulation()

        GRAPH_kernels = Graph_Estimator_Hawkes(estimator_kernel, self.the_update_functions)
        GRAPH_kernels.draw_evolution_parameter_over_time(separator_colour='weight function')
        estimator_kernel.DF.to_csv(trash_path,
                                   index=False,
                                   header=True)




    def test_over_the_time_adaptive_one(self):
        nb_of_times = 50
        width_kernel = 1/5
        b = width_kernel/2

        HAWKSY = Hawkes_process(tt, PARAMETERS)

        estimator_kernel = Estimator_Hawkes()
        # work-in-progress
        #  put optimal kernel here
        my_opt_kernel = Kernel(fct_biweight, name="Biweight", a=-350, b=350)
        Times = np.linspace(0.05 * T, 0.95 * T, nb_of_times)
        ############################## first step
        count_times = 0
        for time in Times:
            count_times += 1
            HAWKSY.update_coef(time, self.the_update_functions, T_max=T)
            print(HAWKSY)
            print("=" * 78)
            print(f"Time : {count_times} out of : {len(Times)}.")

            functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max=T,
                                                        nb_of_guesses=nb_of_guesses,
                                                        kernel_weight=my_opt_kernel, time_estimation=time,
                                                        silent=silent)
        estimator_kernel.DF.to_csv(first_estimation_path,
                                   index=False,
                                   header=True)

    def test_over_the_time_adaptive_two(self):
        # work-in-progress
        #  I got the first estimates. I can potentially already draw the evolution of parameters.
        #  do adaptive here

        # on regarde le estimator_kernel et on en déduit l'optimal bandwidth.

        adaptive_estimator_kernel = Estimator_Hawkes()
        my_adapt_kernel = Kernel_adaptive(fct_biweight, pilot_function_vector=scalings, name="BiWeight", a=-350, b=350)
        ############################## second step
        count_times = 0
        for time in Times:
            count_times += 1
            HAWKSY.update_coef(time, self.the_update_functions, T_max=T)
            print(HAWKSY)
            print("=" * 78)
            print(
                f"Time : {count_times} out of : {len(Times)}."
            )

            functions_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, T_max=T,
                                                        nb_of_guesses=nb_of_guesses,
                                                        kernel_weight=my_opt_kernel, time_estimation=time,
                                                        silent=silent)

        GRAPH_kernels = Graph_Estimator_Hawkes(adaptive_estimator_kernel, self.the_update_functions)
        GRAPH_kernels.draw_evolution_parameter_over_time(separator_colour='weight function')
        adaptive_estimator_kernel.DF.to_csv(
            'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\estimators_adapt.csv',
            index=False,
            header=True)


    def test_change_point_analysis(self):
        functions_change_point_analysis.change_point_plot(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            width=5, min_size=5, n_bkps=1, model="l2", column_for_multi_plot_name='weight function')
