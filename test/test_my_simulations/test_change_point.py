import unittest

from test.test_my_simulations.setup_for_estimations import *
from functions.functions_change_point_analysis import change_point_analysis_and_plot

# section ######################################################################
#  #############################################################################
# setup

if TYPE_ANALYSIS == "optimal":
    parameters_for_analysis = NUMBER_OF_BREAKPOINTS, MODEL, MIN_SIZE

elif TYPE_ANALYSIS == "window":
    parameters_for_analysis = NUMBER_OF_BREAKPOINTS, MODEL, WIDTH


class Test_Simulation_Hawkes_adaptive(unittest.TestCase):
    def tearDown(self):
        plt.show()

    def test_change_point_analysis(self):
        # BIANCA problem path

        print(change_point_analysis_and_plot(
            path=first_estimation_path,
            type_analysis=TYPE_ANALYSIS,
            parameters_for_analysis=parameters_for_analysis,
            true_breakpoints=true_breakpoints,
            column_for_multi_plot_name='weight function'))
        print(change_point_analysis_and_plot(
            path=second_estimation_path,
            type_analysis=TYPE_ANALYSIS,
            parameters_for_analysis=parameters_for_analysis,
            true_breakpoints=true_breakpoints,
            column_for_multi_plot_name='weight function'))
