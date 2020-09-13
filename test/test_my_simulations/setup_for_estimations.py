# normal libraries

# my libraries

# other files
from classes.graphs.class_graph_estimator_hawkes import *
from classes.class_hawkes_process import *
from functions.functions_fct_evol_parameters import update_functions
from test.parameters.parameter_converter_JSON import *

np.random.seed(124)


def choice_parameter(dim, styl):
    # dim chooses how many dimensions
    # styl chooses which variant of the parameters.
    if dim == 1:
        if styl == 1:
            ALPHA = [[1.2]]
            BETA = [[2]]
            MU = [0.2]
            T0, mini_T = 0, 120  # 50 jumps for my uni variate stuff
        elif styl == 2:
            ALPHA = [[2.]]
            BETA = [[2.4]]
            MU = [0.2]
            T0, mini_T = 0, 45  # 50 jumps for my uni variate stuff
        elif styl == 3:
            ALPHA = [[1.75]]
            BETA = [[2]]
            MU = [0.5]
            T0, mini_T = 0, 15  # 50 jumps for my uni variate stuff
        elif styl == 4:
            ALPHA = [[1]]
            BETA = [[4]]
            MU = [0.2]
            T0, mini_T = 0, 45  # 50 jumps for my uni variate stuff


    elif dim == 2:
        if styl == 1:
            ALPHA = [[2, 1],
                     [1, 2]]
            BETA = [[7, 4],
                    [4, 7]]
            MU = [0.2, 0.2]
            T0, mini_T = 0, 70
        elif styl == 2:
            ALPHA = [[2, 2],
                     [1, 2]]
            BETA = [[5, 3],
                    [3, 5]]
            MU = [0.4, 0.3]
            T0, mini_T = 0, 12

    elif dim == 5:
        ALPHA = [[2, 1, 0.5, 0.5, 0.5],
                 [1, 2, 0.5, 0.5, 0.5],
                 [0, 0, 0.5, 0, 0],
                 [0, 0, 0., 0.5, 0.5],
                 [0, 0, 0., 0.5, 0.5]]
        BETA = [[5, 5, 5, 6, 3],
                [5, 5, 5, 6, 3],
                [0, 0, 10, 0, 0],
                [0, 0, 0, 6, 3],
                [0, 0, 0, 6, 3]]
        MU = [0.2, 0.2, 0.2, 0.2, 0.2]
        T0, mini_T = 0, 5

    else:
        raise Error_not_allowed_input("Problem with given dimension")

    ALPHA, BETA, MU = np.array(ALPHA, dtype=np.float), np.array(BETA, dtype=np.float), np.array(MU,
                                                                                                dtype=np.float)  # I precise the type because he might think the np.array is int type.
    PARAMETERS = [MU.copy(), ALPHA.copy(), BETA.copy()]

    print("ALPHA : \n", ALPHA)
    print("BETA : \n", BETA)
    print('NU : \n', MU)
    print("=" * 78)
    print("=" * 78)
    print("=" * 78)
    return PARAMETERS, ALPHA, BETA, MU, T0, mini_T


##########################################
################parameters################
##########################################
# 2000 JUMPS
# T = 200 * mini_T
####################################################################### TIME
# number of max jump
nb_of_sim, M_PREC = 50000, 200000
M_PREC += 1

# section ######################################################################
#  #############################################################################
# simulation

silent = True
test_mode = False

# section ######################################################################
#  #############################################################################
print("\n~~~~~Computations.~~~~~\n")
PARAMETERS, ALPHA, BETA, MU, T0, mini_T = choice_parameter(dim=DIM, styl=STYL)
print(PARAMETERS)
the_update_functions, true_breakpoints = update_functions(FUNCTION_NUMBER, PARAMETERS)

estimator_multi = Estimator_Hawkes()
T_max = NUMBER_OF_MINI_T_IN_SIMULATION * mini_T
# in terms of how many jumps, I want roughly 7500 jumps
# a good precision is 500*(T-T0)

tt = np.linspace(T0, T_max, M_PREC, endpoint=True)

HAWKSY = Hawkes_process(the_update_functions)
# for not keeping the data, I store it in the bin:

trash_path = r"C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files/estimators.csv"
# for the first estimate in the adaptive strategy I store it there:

# BIANCA problem path
first_estimation_path = r"C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\{}".format(
    FILE_ONE)
second_estimation_path = r"C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\{}".format(
    FILE_TWO)
third_estimation_path = r"C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\{}".format(
    FILE_THREE)

# section ######################################################################
#  #############################################################################
# evolution through time


width_kernel = 1 / KERNEL_DIVIDER * T_max
b = width_kernel / 2.
