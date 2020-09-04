from test.test_my_simulations.setup_for_estimations import *
from functions.functions_change_point_analysis import change_point_analysis_and_plot

# section ######################################################################
#  #############################################################################
# setup

type_analysis = "optimal"
number_of_breakpoints = 1
model = "l2"
min_size = 5
width = 5

if type_analysis == "optimal":
    parameters_for_analysis = number_of_breakpoints, model, min_size

elif type_analysis == "window":
    parameters_for_analysis = number_of_breakpoints, model, width

class Test_Simulation_Hawkes_adaptive(unittest.TestCase):
    def tearDown(self):
        plt.show()

    def test_change_point_analysis(self):


        # BIANCA problem path
        print(change_point_analysis_and_plot(
            path = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimations\super_2_second.csv',
            type_analysis = type_analysis,
            parameters_for_analysis = parameters_for_analysis,
            column_for_multi_plot_name='weight function'))

        print(change_point_analysis_and_plot(
            path = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimations\super_2_first.csv',
            type_analysis = type_analysis,
            parameters_for_analysis = parameters_for_analysis,
            column_for_multi_plot_name='weight function'))