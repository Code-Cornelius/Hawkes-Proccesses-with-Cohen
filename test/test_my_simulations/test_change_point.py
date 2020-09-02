from test.test_my_simulations.setup_for_estimations import *

L = 0.02
R = 0.98
h = 2.5
l = width_kernel / T_max / 2


class Test_Simulation_Hawkes_adaptive(unittest.TestCase):
    # section ######################################################################
    #  #############################################################################
    # setup

    def test_change_point_analysis(self):
        functions.functions_change_point_analysis.change_point_plot(
            r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_kernel_mountain_multi.csv',
            width=5, min_size=5, n_bkps=1, model="l2", column_for_multi_plot_name='weight function')
