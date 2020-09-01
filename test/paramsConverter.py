import json

# todo Niels <3 (my cutie) change path
with open('test/params.json') as file:
    json_params = json.load(file)

params = json_params["one"]
FUNCTION_NUMBER = params["function"]
LENGTH = params["length"]
KERNEL_DIVIDER = params["kernel_div"]
L_PARAM = params["L"]
R_PARAM = params["R"]
FILE_ONE = params["name_file_one"]
FILE_TWO = params["name_file_two"]