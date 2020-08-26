##### normal libraries
import time
import unittest

##### my libraries
from library_functions.tools import recurrent_functions

from classes.class_hawkes_process import *
from classes.graphs.class_Graph_Estimator_Hawkes import *
from classes.graphs.class_evolution_plot_estimator_Hawkes import Evolution_plot_estimator_Hawkes
from classes.graphs.class_histogram_estimator_Hawkes import Histogram_estimator_Hawkes
from classes.graphs.class_statistic_plot_estimator_Hawkes import Statistic_plot_estimator_Hawkes
##### other files
from functions import functions_for_MLE
from functions.functions_fct_evol_parameters import update_functions
from .setup_for_estimations import *


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
        intensity, time_real = HAWKSY.simulation_Hawkes_exact_with_burn_in(tt=tt, plot_bool=True, silent=False)
        HAWKSY.plot_hawkes(tt, time_real, intensity, name="EXACT_HAWKES")

    def test_simple_unique(self):
        _, time_real = HAWKSY.simulation_Hawkes_exact_with_burn_in(tt=tt, plot_bool=False, silent=True)
        print(len(time_real[0]))
        print(functions_for_MLE.call_newton_raph_MLE_opt(time_real, T_max, silent=False))

    def test_from_csv(self):
        hist_test = Histogram_estimator_Hawkes.from_path(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimation\super_0_first.csv',
            the_update_functions)

        stat_test = Statistic_plot_estimator_Hawkes.from_path(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimation\super_0_first.csv',
            the_update_functions)

        evol_test = Evolution_plot_estimator_Hawkes.from_path(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimation\super_0_first.csv',
            the_update_functions)

        hist_test.draw()

        TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T]
        stat_test.draw(mini_T=mini_T, times=TIMES, name_column_evolution='T_max',
                       computation_function=recurrent_functions.compute_MSE, class_for_hist=Histogram_estimator_Hawkes,
                       fct_parameters=the_update_functions)  # last parameter for hist.
        evol_test.draw(separator_colour='weight function')

    def test_simple_estimation_multi(self):
        estimator_multi = Estimator_Hawkes()
        functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_multi, tt, nb_of_guesses, silent=silent)
        hist_test = Histogram_estimator_Hawkes(estimator_multi, the_update_functions)
        hist_test.draw()
        estimator_multi.to_csv(trash_path, index=False, header=True)

    def test_MSE(self):
        estimator_MSE = Estimator_Hawkes()
        if test_mode:
            TIMES = [5 * mini_T, 10 * mini_T, 15 * mini_T]
        else:
            TIMES = [10 * mini_T, 20 * mini_T, 30 * mini_T, 40 * mini_T, 45 * mini_T,
                     50 * mini_T, 60 * mini_T, 75 * mini_T, 90 * mini_T, 100 * mini_T, 120 * mini_T,
                     130 * mini_T, 140 * mini_T, 150 * mini_T, 160 * mini_T]
        count_times = 0
        for times in TIMES:
            tt = np.linspace(T0, times, M_PREC, endpoint=True)
            count_times += 1
            print("=" * 78)
            print(
                f"Time : {count_times} out of : {len(TIMES)}.")
            functions_for_MLE.multi_estimations_at_one_time(HAWKSY, estimator_MSE,
                                                            tt, nb_of_guesses, silent=silent)
        stat_test = Statistic_plot_estimator_Hawkes(estimator_MSE, the_update_functions)
        stat_test.draw(mini_T=mini_T, times=TIMES, name_column_evolution='T_max',
                       computation_function=recurrent_functions.compute_MSE, class_for_hist=Histogram_estimator_Hawkes,
                       fct_parameters=the_update_functions)  # last parameter for hist.
        estimator_MSE.DF.to_csv(trash_path, index=False,
                                header=True)

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

        aplot = class_aplot.APlot(how=(1, 1))
        aplot.uni_plot(nb_ax=0, xx=T_plot, yy=vect_time_simulation)
        aplot.set_dict_fig(0, {
            'title': "Increase in time for simulation and convergence of the estimation for Hawkes processes, "
                     "batches of {} realisations.".format(
                nb_of_guesses),
            'xlabel': "Number of Jumps simulated", 'ylabel': "Average time to simulate"})
        aplot.save_plot("Timing_opti")
        return
