import json

# BIANCA problem path

path_file_simulation = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters\parameters_for_simulation.json'
path_file_paths = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters\parameters_for_files.json'
path_file_adaptive_simulation = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters\parameters_for_adaptive_estimation.json'
path_file_change_point = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters\parameters_for_change_point_analysis.json'

with open(path_file_simulation) as file1:
    json_parameters_simulation = json.load(file1)
with open(path_file_paths) as file2:
    json_parameters_paths = json.load(file2)
with open(path_file_adaptive_simulation) as file3:
    json_parameters_adaptive_simulation = json.load(file3)
with open(path_file_change_point) as file4:
    json_parameters_change_point = json.load(file4)

the_json_parameters_simulation = json_parameters_simulation["basic_config"]
the_json_parameters_paths = json_parameters_paths["basic_config"]
the_json_parameters_adaptive_simulation = json_parameters_adaptive_simulation["basic_config"]
the_json_parameters_change_point = json_parameters_change_point["basic_config"]

# section ######################################################################
#  #############################################################################
# initialization

FUNCTION_NUMBER = the_json_parameters_simulation["function"]
NUMBER_OF_MINI_T_IN_SIMULATION = the_json_parameters_simulation["number_of_mini_t_in_simulation"]
KERNEL_DIVIDER = the_json_parameters_simulation["kernel_div"]
NB_OF_TIMES = the_json_parameters_simulation["nb_of_times"]
DIM = the_json_parameters_simulation["dim"]
STYL = the_json_parameters_simulation["styl"]
NB_OF_GUESSES = the_json_parameters_simulation["nb_of_guesses"]

FILE_ONE = the_json_parameters_paths["name_file_one"]
FILE_TWO = the_json_parameters_paths["name_file_two"]
FILE_THREE = the_json_parameters_paths["name_file_three"]

L_PARAM = the_json_parameters_adaptive_simulation["L"]
R_PARAM = the_json_parameters_adaptive_simulation["R"]
h_PARAM = the_json_parameters_adaptive_simulation["h"]
l_PARAM = the_json_parameters_adaptive_simulation["l"]
CONSIDERED_PARAM = the_json_parameters_adaptive_simulation["considered_parameters"]
ALL_KERNELS_DRAWN = the_json_parameters_adaptive_simulation["all_kernels_drawn"]

TYPE_ANALYSIS = the_json_parameters_change_point["type_analysis"]
NUMBER_OF_BREAKPOINTS = the_json_parameters_change_point["number_of_breakpoints"]
MODEL = the_json_parameters_change_point["model"]
MIN_SIZE = the_json_parameters_change_point["min_size"]
WIDTH = the_json_parameters_change_point["width"]
