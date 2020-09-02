"""
import csv
import numpy as np
from classical_functions import *
from functions_networkx import *

from functions import functions_for_MLE


############# tests ####################
alpha = 3


start = time.time()
for j in range( 1000 ):
    my_list = np.array(range(3000))/1000
    ans = sum( function_vanilla_option_output(x, alpha) for x in my_list * alpha)
print( "time duration : ", time.time() - start )


import math

start = time.time()
for j in range( 100 ):
    my_list = np.array(range(3000))/1000
    ans = sum( math.exp(x) for x in my_list*alpha)
print( "time duration math : ", time.time() - start )


start = time.time()
for j in range( 100 ):
    my_list = np.array(range(3000))/1000
    my_list = np.exp(my_list*alpha)
    ans = sum( my_list)
print( "time duration outside : ", time.time() - start )




##########################################
################parameters################
##########################################
# number of max jump
nb_of_sim = 50000
# timing
T0 = 0
mini_T = 35 # 50 jumps
# so here I should have around 500 jumps.
#T = 1 * mini_T
# 2000 JUMPS
T = 100 * mini_T

####################################################################### TIME
M_PREC = 150000;
M_PREC += 1
# a good precision is 500*(T-T0)


tt = np.linspace(T0, T, M_PREC, endpoint=True)


ALPHA = [[1.75]]
BETA = [[2]]
MU = [0.2]

ALPHA = np.array(ALPHA);
BETA = np.array(BETA);
MU = np.array(MU)
print("ALPHA : \n ", ALPHA)
print("BETA : \n ", BETA)
print('MU : \n ', MU)
print("=" * 78)
print("=" * 78)
print("=" * 78)
np.random.seed(124)


# Simulation writing in csv


##########################################
###################main###################
##########################################
#################### Exact simulation of multidimensional Hawkes
intensity, time_real = simulation_Hawkes_exact_with_burn_in(tt, ALPHA, BETA, MU, nb_of_sim=nb_of_sim, T_max=T, plot_bool = False)



zippable = zip(*time_real)
myFile = open('jumps_hawkes.csv', 'w', newline = '')
with myFile:
   writer = csv.writer(myFile)
   writer.writerows(zippable)




# Using csv file
myFile = open('jumps_hawkes.csv', 'r', newline = '')
reader = csv.reader(myFile)

time_real = [[]]
for row in reader:
    for e in row:
        time_real[0].append( float(e) )


# normal estimation
my_time = time.time()
print(functions_for_MLE.call_newton_raph_MLE_opt(time_real, T))
time_computational(my_time, time.time(), "my function")



# other method

import derivatives_MLE
M = 1

f   = lambda x: derivatives_MLE.likelihood(time_real,
                                            np.reshape(x[:M*M], (M,M)  ),
                                            np.reshape(x[M*M:2*M*M], (M,M)  ),
                                            x[2* M*M: ],
                                           T = T)
df  = lambda x: derivatives_MLE.first_derivative(time_real,
                                            np.reshape(x[:M*M], (M,M)  ),
                                            np.reshape(x[M*M:2*M*M], (M,M)  ),
                                            x[2* M*M: ],
                                           T = T)
ddf = lambda x: derivatives_MLE.second_derivative(time_real,
                                            np.reshape(x[:M*M], (M,M)  ),
                                            np.reshape(x[M*M:2*M*M], (M,M)  ),
                                            x[2* M*M: ],
                                           T = T)

my_time = time.time()
x = [1.5,2.2,0.1]
print(x)
print(scipy.optimize.minimize(f, x, method='Newton-CG',
                        jac=df,
                        hess=ddf )
)
time_computational(my_time, time.time(), "scipy function NEWTON CONJUGATE GRADIENT")




# one_long_and_longer_estimation(tt, ALPHA, BETA, MU, mini_T)
# print(multi_simul_Hawkes_and_estimation(tt, ALPHA, BETA, MU, T, nb_of_guesses = 300))
# multi_simul_Hawkes_and_estimation_MSE(tt, ALPHA, BETA, MU, mini_T = mini_T, nb_of_guesses= 100)
plt.show()

"""
