import json
#BIANCA problem path
with open(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\test\parameters_simulation.json') as file:
    json_params = json.load(file)

params = json_params["one"]
FUNCTION_NUMBER = params["function"]
NUMBER_OF_MINI_T_IN_SIMULATION = params["number_of_mini_t_in_simulation"]
KERNEL_DIVIDER = params["kernel_div"]
L_PARAM = params["L"]
R_PARAM = params["R"]
FILE_ONE = params["name_file_one"]
FILE_TWO = params["name_file_two"]
FILE_THREE = params["name_file_three"]
CONSIDERED_PARAM = params["considered_parameters"]