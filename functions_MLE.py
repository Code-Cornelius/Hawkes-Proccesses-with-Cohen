import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import networkx as nx
from inspect import signature

# from sys import path
# path.append("C:\Users\nie_k\Desktop\travail\EPFL 3eme ANNEE")
from generic_functions import *
from plot_functions import *
from useful_functions import *
from functions_networkx import *

from class_kernel import *
from class_hawkes_process import *

from functions_derivatives_MLE import *


# e is the error.
# tol is the each step tolerance
# An interesting number of jump for that algorithm is around a hundred.
def newtons_method_multi_MLE(df, ddf, ALPHA, BETA, MU, e=10 ** (-10), tol=3 * 10 ** (-4), silent=True):
    # nb of dimensions
    M = len(MU)
    ## df is the derivative
    ## ddf its Hessian
    ## x0 first guess
    ## e the tolerance
    # while f is bigger than the tolerance.
    number_of_step_crash = 0
    step = 0  # first change in norm
    multi = 0.5
    b = 0.1
    # that number is for saying that if it explodes too many times, just leave that realisation out.
    nb_of_explosions = 0
    changed = False

    # this is to know if we reached a point where there is huge movement, so starting from that index, we re initialize the multi coefficient.
    reset_index = 0
    derivative = df(ALPHA, BETA, MU)
    while np.linalg.norm(derivative, 2) > e and step > tol or number_of_step_crash == 0:  # I use norm 2 as criterea
        # Printing
        if not silent:
            if number_of_step_crash % 10 == 0:
                print("Step Newton {} result {}, change in norm {}.".format(number_of_step_crash,
                                                                            np.linalg.norm(derivative, 2), step))
                print(
                    "                                             GUESSES : \n ALPHA : \n {}  \n BETA : \n {} \n NU : \n {}".format(
                        ALPHA, BETA, MU))
        if number_of_step_crash > 1500:
            raise Exception("Is the function flat enough ?")
        number_of_step_crash += 1

        # stock the old guess for update
        old_x0 = np.append(np.append(MU, np.ravel(ALPHA)), np.ravel(BETA))  # the ravel flattens the matrix

        # compute the shift
        hessian = ddf(ALPHA, BETA, MU)
        # if not invertible you re do the simulations. Solve is also more effective than computing the inverse
        if not is_invertible(hessian):
            return False, 1, 1, 1
        direction = np.linalg.solve(hessian, derivative)

        # test :print(previous_Rs);;print(previous_Rs_dash_dash);print(previous_denomR)
        # new coefficient armijo rule if the iterations are very high.
        # the conditions are first
        # 1. if we are at the beg of the iterations and convergence
        # 2. if we are not too close yet of the objective, the derivative equal to 0. One scale by M bc more coeff implies bigger derivative
        # 3. nb of explosions, if there are explosions it means I need to be more gentle to find the objective
        if number_of_step_crash - reset_index < 10 and np.linalg.norm(derivative,
                                                                      2) > 50 * M * M and nb_of_explosions < 2:
            multi = 1 / M ** 3
        elif number_of_step_crash - reset_index < 10 and np.linalg.norm(derivative,
                                                                        2) > 2 * M * M and nb_of_explosions < 5:
            multi = 0.6 / M ** 3
        elif number_of_step_crash - reset_index < 100 and np.linalg.norm(derivative, 2) > 0.01 * M * M:
            multi = 0.2 / M ** 3
        elif number_of_step_crash < 500:  # and np.linalg.norm(derivative, 2) > 0.1*M:
            multi = 0.05 / M ** 3
        elif number_of_step_crash < 1200:
            variable_in_armijo = ALPHA, BETA, MU
            multi, changed = armijo_rule(df, ddf, variable_in_armijo, direction, a=multi, sigma=0.5, b=b)
        # else :
        # the else is already handled at the beginning.    break

        # new position
        x0 = old_x0 - multi * direction

        # if the coefficient given by armijo is too small I change it.
        if multi < 10e-8:
            changed = False
            multi = 10e-3

        #IF armijo was applied,
        # in order to still got some loose when moving on the derivatives, I divide by the coef
        if changed:
            multi /= b

        # if the max is too big I replace the value by a random number between 0 and 1.
        # Also, I synchronize the alpha and the beta in order to avoid boundary problem.
        if np.max(x0) > 100:
            nb_of_explosions += 1
            for i in range(len(x0)):
                if i < M:
                    x0[i] = np.random.rand(1)
                elif i < M + M * M:
                    random_value = np.random.rand(1)
                    x0[i] = random_value
                    x0[i + M * M] = 2 * M * M * random_value
                # The list is already full, break.
                else:
                    break
            # I reset the step size.
            reset_index = number_of_step_crash

        # Here I deal with negative points
        for i in range(len(x0)):
            if i >= M + M * M:  # betas, they can't be negative otherwise overflow in many expressions involving exponentials.
                if x0[i] < 0:
                    x0[i] = 0.1
            elif x0[i] < -0.01:
                x0[i] = - x0[i]

        # In order to avoid infinite loops, I check if there was too many blow ups. If there are too many, I return flag as false.
        if nb_of_explosions > 10:
            return False, 1, 1, 1

        # normally step won't be used. It is a dummy variable "moved or not". (under Armijo)
        step = np.linalg.norm(x0 - old_x0, 2)
        # get back the guesses
        MU = x0[:M]
        ALPHA = np.reshape(x0[M:M * M + M], (M, M))  # matrix shape
        BETA = np.reshape(x0[M * M + M:], (M, M))

        # big changes, reset the multi index. Makes explosion faster. The Steps shouldn't be that big.
        if step > 5:
            reset_index = number_of_step_crash

        # reduces some computations to put it here
        derivative = df(ALPHA, BETA, MU)

    # True because it was successful.
    return True, ALPHA, BETA, MU



























#return 3 things, first the coefficient by which to multiply the stepest descent.
# also which direction has to change.
# finally whether the coefficient has been changed.
def armijo_rule(f, df, x0, direction, a, sigma, b):
    #TODO ARMIJO RULE IS DONE FOR CASES WHERE ALPHA BETAM U ARE SCALARS, MULTIVARIATE CASE!!!
    if abs(b) >= 1:
        raise Exception("b has to be smaller than 1.")

    ALPHA, BETA, MU = x0
    M = len(ALPHA)

    changed = False
    vector_limit_sup = np.matmul(df(ALPHA, BETA, MU), direction)
    condition = (f(ALPHA + a * direction[:M], BETA + a * direction[M:2 * M],
                   MU + a * direction[2 * M:]) - f(ALPHA, BETA, MU) <= sigma * a * vector_limit_sup)

    # I put .all, I only update if every dimension helps improving.
    # a > 10e-1O in order to not have a too small step.
    while not condition.all() and a > 10e-10:
        a *= b
        changed = True
        condition = (f(ALPHA + a * direction[:M], BETA + a * direction[M:2 * M],
                       MU + a * direction[2 * M:]) - f(ALPHA, BETA, MU) <= sigma * a * vector_limit_sup)

    # print( "limit : ",  sigma * a * vector_limit_sup )
    # print( "value : ", f(ALPHA + a * direction[:M], BETA + a * direction[M:2 * M], MU + a * direction[2 * M:]) - f(ALPHA, BETA, MU)  )
    print("we are in ARMIJO condition because too many steps, the ok directions and step :" + str(
        condition) + " and " + str(a))
    print("derivatives value : ", f(ALPHA, BETA, MU))
    return a, changed


# the function returns a flag for the reason beeing that if it failed to converge too many times, it s perhaps better to try on a new data set.
def call_newton_raph_MLE_opt(T_t, T, w=None, silent=True):
    # T_t = [[6,7,8],[3,7.5,9]]

    if w is None:
        w = Kernel(fct_plain, "plain").eval(T_t, 0) # eval point equals 0 because, if the weights haven't been defined earlier, it means we don't care when we estimate.

    M = len(T_t)
    MU = np.full(M, 0.1)
    ALPHA = np.full((M,M), 0.7)
    BETA = 0.2 + 1.1 * M*M * ALPHA

    #ALPHA = np.array([[10, 0], [0, 10]]) *0.9
    #BETA = np.array([[50, 10], [30, 30]]) * 0.9
    #MU = np.array([1, 1]) * 0.9

    df = lambda ALPHA, BETA, MU: first_derivative(T_t, ALPHA, BETA, MU, T, w)
    ddf = lambda ALPHA, BETA, MU: second_derivative(T_t, ALPHA, BETA, MU, T, w)


    flag, ALPHA, BETA, MU = newtons_method_multi_MLE(df, ddf, ALPHA, BETA, MU, silent = silent)
    return flag, ALPHA, BETA, MU



def estimation_hp(hp, estimator, T_max, nb_of_guesses, kernel_weight = kernel_plain, time_estimation=0, silent = True): #BIANCA-HERE (*) BIANCA a better way to do that?
    ## function_weight should be ONE kernel from class_kernel.
    ## hp is a hawkes process
    ## the flag notes if the convergence was a success. If yes, function hands in the results

    # BIANCA-HERE (*) BIANCA a better way to do that?
    ## the kwargs is any additionnal parameter I need to give to the dataframe.
    flag_success_convergence = False
    while not flag_success_convergence:
        intensity, time_real = hp.simulation_Hawkes_exact(T_max=T_max, plot_bool=False, silent=True)
        w = kernel_weight.eval(T_t=time_real, eval_point=time_estimation)
        flag_success_convergence, alpha_hat, beta_hat, mu_hat = functions_MLE.call_newton_raph_MLE_opt(time_real, T_max,
                                                                                                       w, silent=silent)
    _, M = np.shape(alpha_hat)
    for s in range(M):
        estimator.DF = (estimator.DF).append(pd.DataFrame(
            {"time estimation": [time_estimation],
             "variable": ["nu"],
             "n": [s],
             "m": [0],
             "weight function": [kernel_weight.name],
             "value": [mu_hat[s]],
             'T_max': [T_max],
             'true value': [hp.NU[s]],
             'number of guesses': [nb_of_guesses]  # BIANCA-HERE (*) BIANCA a better way to do that?
             }), sort=True
        )
        for t in range(M):
            estimator.DF = (estimator.DF).append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "variable": ["alpha"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [alpha_hat[s, t]],
                 'T_max': [T_max],
                 'true value': [hp.ALPHA[s, t]],
                 'number of guesses': [nb_of_guesses]  # BIANCA-HERE (*) BIANCA a better way to do that?
                 }), sort=True
            )
            estimator.DF = (estimator.DF).append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "variable": ["beta"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [beta_hat[s, t]],
                 'T_max': [T_max],
                 'true value': [hp.BETA[s, t]],
                 'number of guesses': [nb_of_guesses]  # BIANCA-HERE (*) BIANCA a better way to do that?
                 }), sort=True
            )
    return  # no need to return the estimator.


# we want to run the same simulations a few number of times and estimate the Hawkes processes' parameters every time.
# the length of simulation is given by T
def multi_estimations_at_one_time(hp, estimator, T_max, nb_of_guesses, kernel_weight=kernel_plain, time_estimation=0,
                                  silent=False):
    for i in range(nb_of_guesses):
        if not silent:
            if i % 1 == 0:
                print("estimation {} out of {} estimations.".format(i, nb_of_guesses))
        else:
            if i % 20 == 0:
                print("estimation {} out of {} estimations.".format(i, nb_of_guesses))
        # BIANCA-HERE (*) BIANCA a better way to do that?
        estimation_hp(hp, estimator, T_max, kernel_weight=kernel_weight, time_estimation=time_estimation, silent=silent,
                      nb_of_guesses=nb_of_guesses)

    return # no need to return the estimator.























# those two functions are quite useless right now. I don't want to spend time on them to refractor them.

'''
def capabilities_test_optimization(tt, ALPHA, BETA, MU, mini_T):
    nb_of_trial_for_timing = 50
    T = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T, 40 * mini_T,
         50 * mini_T, 60 * mini_T, 70 * mini_T, 80 * mini_T, 90 * mini_T, 100 * mini_T, 120 * mini_T]
    # test
    # T = [mini_T, 2 * mini_T, 4 * mini_T, 5 * mini_T]
    T_plot = [T[i] // mini_T * 50 for i in range(len(T))]
    vect_time_simulation = np.zeros(len(T))
    # TODO with estimateur_variance(x_real, mean) estimate variance of each parameter

    process = Hawkes_process(tt, ALPHA, BETA, MU)
    for i_times in range(len(T)):
        time_simulation = 0
        # histogram for last value of T

        print("=" * 78)
        print("=" * 78)
        print("=" * 78)

        for i in range(nb_of_trial_for_timing):
            print("=" * 78)
            print("Global timing simulation : Big time {}, out of {}, successive simulation {} out of {}.".format(
                i_times + 1, len(T), i + 1, nb_of_trial_for_timing))
            print("=" * 78)

            start = time.time()
            process.gen_plus_conv_hawkes(T[i_times])
            time_simulation += time.time() - start

        vect_time_simulation[i_times] = time_simulation / nb_of_trial_for_timing
    plot_functions.plot_graph(T_plot, vect_time_simulation, title=[
        "Increase in time for simulation and convergence of the estimation for Hawkes processes, batches of {} realisations.".format(
            nb_of_trial_for_timing)],
                              labels=["Number of Jumps simulated", "Time"],
                              parameters=[ALPHA, BETA, MU], name_parameters=["ALPHA", "BETA", "NU"],
                              name_save_file="Timing_opti")
    return



def one_long_and_longer_estimation(tt, ALPHA, BETA, MU, mini_T):
    _, M = np.shape(ALPHA)
    # my final guesses

    # mini_T should represent a time interval of roughly 50 jumps.
    T = [5 * mini_T, 10 * mini_T, 15 * mini_T, 20 * mini_T, 25 * mini_T, 30 * mini_T, 35 * mini_T, 40 * mini_T,
         45 * mini_T,
         50 * mini_T, 55 * mini_T, 60 * mini_T, 65 * mini_T, 75 * mini_T, 80 * mini_T, 90 * mini_T,
         100 * mini_T, 110 * mini_T, 125 * mini_T, 150 * mini_T, 175 * mini_T, 200 * mini_T,
         220 * mini_T, 250 * mini_T, 275 * mini_T, 300 * mini_T, 400 * mini_T]
    # test
    # T = [2 * mini_T, 3 * mini_T, 4 * mini_T]
    list_of_succesive_coefficients = np.zeros((len(T), M + 2 * M * M))

    # I can't call the function gen_plus_conv_hawkes so be careful using the function.
    intensity, time_real = simulation_Hawkes_exact(tt, ALPHA, BETA, MU,
                                                   T_max=T[-1], plot_bool=False)
    actual_times = [[] for i in range(M)]
    for i in range(len(T)):
        print("=" * 78)
        print("One longer simulation : Step {} out of {}.".format(i + 1, len(T)))
        indeces = find_smallest_rank_leq_to_K(np.array(time_real), T[i], sorted=True)
        for j in range(M):
            # TODO one can optimize this by avoiding repetition of the extension of stuff already put in (extend)
            actual_times[j] = time_real[j][:indeces[j]]
        # print("debug : ", actual_times)
        flag_success, alpha_hat, beta_hat, mu_hat = functions_MLE.call_newton_raph_MLE_opt(actual_times, T[i])
        if flag_success:  # means the algo converged. If didn't, I leave the spaces as 0.
            for j in range(M):
                list_of_succesive_coefficients[i][j] = mu_hat[j]
            for j in range(M * M):
                list_of_succesive_coefficients[i][j + M] = alpha_hat[j]
                list_of_succesive_coefficients[i][j + M + M * M] = beta_hat[j]

    # print(list_of_succesive_coefficients)
    # plot the graphs for every coefficient
    for i in range(len(list_of_succesive_coefficients[0, :])):
        if i < M:
            title = " Evolution of the estimation of the estimator of the parameter NU, {}th value of the vector.".format(
                i + 1)
            y = np.full(len(T), MU[i])
            save_name = "evolution_time_mu_{}".format(i + 1)
        elif i < M + M * M:
            title = " Evolution of the estimation of the parameter ALPHA, {}th value of the vector.".format(i + 1 - M)
            save_name = "evolution_time_alpha_{}".format(i + 1 - M)
            y = np.full(len(T), np.ravel(ALPHA)[i - M])
        else:
            title = " Evolution of the estimation of the parameter BETA, {}th value of the vector.".format(
                i + 1 - M - M * M)
            y = np.full(len(T), np.ravel(BETA)[i - M - M * M])
            save_name = "evolution_time_beta_{}".format(i + 1 - M - M * M)
        plot_functions.plot_graph(T, list_of_succesive_coefficients[:, i], title=[title],
                                  labels=["Time", "Value of the Estimator"],
                                  parameters=[ALPHA, BETA, MU], name_parameters=["ALPHA", "BETA", "NU"],
                                  name_save_file=save_name)
        plt.plot(T, y, color='blue', marker="None", linestyle='dashed', linewidth=2)
    return alpha_hat, beta_hat, mu_hat

'''