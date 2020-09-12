import json
#BIANCA problem path
with open(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters_simulation.json') as file:
    json_params = json.load(file)

the_json_parameters = json_params["one"]
FUNCTION_NUMBER = the_json_parameters["function"]
NUMBER_OF_MINI_T_IN_SIMULATION = the_json_parameters["number_of_mini_t_in_simulation"]
KERNEL_DIVIDER = the_json_parameters["kernel_div"]
L_PARAM = the_json_parameters["L"]
R_PARAM = the_json_parameters["R"]
h_PARAM = the_json_parameters["h"]
l_PARAM = the_json_parameters["l"]
FILE_ONE = the_json_parameters["name_file_one"]
FILE_TWO = the_json_parameters["name_file_two"]
FILE_THREE = the_json_parameters["name_file_three"]
CONSIDERED_PARAM = the_json_parameters["considered_parameters"]
ALL_KERNELS_DRAWN = the_json_parameters["all_kernels_drawn"]

TYPE_ANALYSIS =the_json_parameters["type_analysis"]
NUMBER_OF_BREAKPOINTS =the_json_parameters["number_of_breakpoints"]
MODEL =the_json_parameters["model"]
MIN_SIZE = the_json_parameters["min_size"]
WIDTH = the_json_parameters["width"]
