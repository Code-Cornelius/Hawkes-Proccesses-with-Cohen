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

        width_kernel = 1 / 5. * T
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

        Times = np.linspace(0.05 * T, 0.95 * T, nb_of_times)

        total_nb_tries = len(Times) * len(list_of_kernels)
        actual_state = [0]  # initialization

        if to_be_simulated:
            @decorators_functions.prediction_total_time(total_nb_tries=total_nb_tries,
                                                        multiplicator_factor=0.9,
                                                        actual_state=actual_state)
            def simulation():
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
                print(HAWKSY(a_time, T))
                for kernel in list_of_kernels:
                    count_kernels += 1
                    actual_state[0] += 1
                    simulation()
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
        to_be_simulated = False
        path = first_estimation_path
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\super_3_first.csv'


        if test_mode:
            nb_of_times = 3
        else:
            nb_of_times = 50

        width_kernel = 1 / 5 * T
        b = width_kernel / 2

        estimator_kernel = Estimator_Hawkes()
        #  put optimal kernel here
        my_opt_kernel = Kernel(fct_biweight, name=f"Biweight {width_kernel} width", a=-b, b=b)
        Times = np.linspace(0.05 * T, 0.95 * T, nb_of_times)
        actual_state = [0]  # initialization

        if to_be_simulated:
            @decorators_functions.prediction_total_time(total_nb_tries=len(Times),
                                                        multiplicator_factor=0.9,
                                                        actual_state=actual_state)
            def simulation():
                print(''.join(["\n", "=" * 78]))
                print(f"Time : {count_times} out of : {len(Times)}.")
                functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_kernel, tt=tt,
                                                                nb_of_guesses=nb_of_guesses,
                                                                kernel_weight=my_opt_kernel, time_estimation=a_time,
                                                                silent=silent)


            ############################## first step
            count_times = 0
            for a_time in Times:
                print(HAWKSY(a_time, T))
                count_times += 1
                actual_state[0] += 1
                simulation()

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
        path = 'C:\\Users\\nie_k\\Desktop\\travail\\RESEARCH\\RESEARCH COHEN\\super_0_first.csv'

        considered_param = ['nu','alpha','beta']

        if test_mode:
            nb_of_times = 3
        else:
            nb_of_times = 50

        width_kernel = 1 / 5. * T
        b = width_kernel / 2.
        Times = np.linspace(0.05 * T, 0.95 * T, nb_of_times)
        estimator_kernel = Estimator_Hawkes.from_path(path)

        # by looking at the previous estimation, we deduce the scaling
        # for that I take back the estimate
        # there is a probelm of data compatibility, so I put the keys as integers, assuming that there is no estimation on the same integer.
        my_estimator_dict = estimator_kernel.mean(separator='time estimation') #take back the value of the estimation at a given time.
        my_estimator_dict = {int(key): my_estimator_dict[key] for key in my_estimator_dict.keys() }
        list_of_estimation = []
        # mean returns a dict, so I create my list of list:
        for a_time in Times:
            a_time = int(a_time)
            list_of_estimation.append(my_estimator_dict[a_time])

        my_scaling = functions_fct_rescale_adaptive.rescaling_kernel_processing(Times, list_of_estimation, considered_param)
        print('the scaling : ', my_scaling)
        # the kernel is taken as biweight.
        list_of_kernels = functions_fct_rescale_adaptive.creator_list_kernels(my_scaling, b)

        adaptive_estimator_kernel = Estimator_Hawkes()
        actual_state = [0]  # initialization

        @decorators_functions.prediction_total_time(total_nb_tries=len(Times),
                                                    multiplicator_factor=0.9,
                                                    actual_state=actual_state)
        def simulation(a_time, kernel):
            print(''.join(["\n", "=" * 78]))
            print(f"Time : {count_times} out of : {len(Times)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, adaptive_estimator_kernel, tt=tt,
                                                            nb_of_guesses=nb_of_guesses,
                                                            kernel_weight=kernel, time_estimation=a_time,
                                                            silent=silent)

        ############################## second step
        count_times = 0

        # for a_time, kernel in zip(Times, list_of_kernels):
        #     HAWKSY.update_coef(a_time, self.the_update_functions, T_max=T)
        #     print(HAWKSY)
        #     count_times += 1
        #     actual_state[0] += 1
        #     simulation(a_time, kernel)

        evol_kernels = Evolution_plot_estimator_Hawkes(estimator_kernel, the_update_functions)
        plot_param = list_of_kernels, Times
        evol_kernels.draw(separator_colour='weight function', kernel_plot_param=plot_param)
        estimator_kernel.to_csv(second_estimation_path, index=False, header=True)











    def test_change_point_analysis(self):
        functions_change_point_analysis.change_point_plot(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            width=5, min_size=5, n_bkps=1, model="l2", column_for_multi_plot_name='weight function')