# normal libraries
import warnings
import pandas as pd

##### my libraries
from errors.Error_convergence import *

from classes.class_hawkes_process import *
from functions_derivatives_MLE import *
##### other files
from functions_my_newton_raphson import newtons_method_multi_MLE


# the function returns a flag for the reason beeing that if it failed to converge too many times, it s perhaps better to try on a new data set.
# now there is an error raised.
def simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation):
    T_max = tt[-1]
    _, time_real = hp.simulation_Hawkes_exact_with_burn_in(tt=tt, plot_bool=False, silent=True) # don't store intensity, only used for plots.
    w = kernel_weight.eval(T_t=time_real, eval_point=time_estimation, T_max = T_max)
    # print(time_real)
    try:
        alpha_hat, beta_hat, mu_hat = call_newton_raph_MLE_opt(time_real, T_max, w, silent=silent)
    except Error_convergence as err:
        warnings.warn(err.message)
        return simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation)
    # One shouldn't get an infinite loop here. It's probability.
    return alpha_hat, beta_hat, mu_hat

def call_newton_raph_MLE_opt(T_t, T, w=None, silent=True):
    # w shouldn't be None, however as a safety measure, just before doing the computations !
    if w is None:
        w = Kernel(fct_plain, "plain", T_max =  T).eval(T_t, 0, T_max = T)
        # eval point equals 0 because, if the weights haven't been defined earlier, it means we don't care when we estimate.
    M = len(T_t)
    MU = np.full(M, 0.1)
    ALPHA = np.full((M, M), 0.7)
    BETA = 0.2 + 1.1 * M * M * ALPHA

    # ALPHA = np.array([[2, 1], [1, 2]]) * 0.99
    # BETA = np.array([[5, 3], [3, 5]]) * 0.99
    # MU = np.array([0.2, 0.2]) * 0.99

    # ALPHA = np.array([[1, 2], [1, 2]])
    # BETA = np.array([[5, 10], [5, 10]])
    # MU = np.array([1, 1])

    # print("debug")
    # T = 20
    # print(T_t)
    # T_t = [[17, 18, 19], [16, 19]]
    # w = Kernel(fct_plain, "plain").eval(T_t, 0)
    # print("debug")

    df = lambda MU, ALPHA, BETA: first_derivative(T_t, ALPHA, BETA, MU, T, w)
    ddf = lambda MU, ALPHA, BETA: second_derivative(T_t, ALPHA, BETA, MU, T, w)

    MU, ALPHA, BETA = newtons_method_multi_MLE(df, ddf, ALPHA, BETA, MU, silent=silent)
    return ALPHA, BETA, MU


def estimation_hp(hp, estimator, tt, nb_of_guesses, kernel_weight= kernel_plain, time_estimation=0,
                  silent=True):
    ## function_weight should be ONE kernel from class_kernel.
    ## hp is a hawkes process
    ## the flag notes if the convergence was a success. If yes, function hands in the results

    alpha_hat, beta_hat, mu_hat = simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation)

    _, M = np.shape(alpha_hat)
    T_max = tt[-1]
    for s in range(M):
        estimator.DF = (estimator.DF).append(pd.DataFrame(
            {"time estimation": [time_estimation],
             "parameter": ["nu"],
             "n": [s],
             "m": [0],
             "weight function": [kernel_weight.name],
             "value": [mu_hat[s]],
             'T_max': [T_max],
             'time_burn_in': [Hawkes_process.time_burn_in],
             'true value': [hp.NU[s](time_estimation,T_max, Hawkes_process.time_burn_in)],
             'number of guesses': [nb_of_guesses]
             }), sort=True
        )
        for t in range(M):
            estimator.DF = (estimator.DF).append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "parameter": ["alpha"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [alpha_hat[s, t]],
                 'T_max': [T_max],
                 'time_burn_in': [Hawkes_process.time_burn_in],
                 'true value': [hp.ALPHA[s][t](time_estimation,T_max, Hawkes_process.time_burn_in) ],
                 'number of guesses': [nb_of_guesses]
                 }), sort=True
            )
            estimator.DF = (estimator.DF).append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "parameter": ["beta"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [beta_hat[s, t]],
                 'T_max': [T_max],
                 'time_burn_in': [Hawkes_process.time_burn_in],
                 'true value': [hp.BETA[s][t](time_estimation,T_max, Hawkes_process.time_burn_in)],
                 'number of guesses': [nb_of_guesses]
                 }), sort=True
            )
    return  # no need to return the estimator.


# we want to run the same simulations a few number of times and estimate the Hawkes processes' parameters every time.
# the length of simulation is given by T
def multi_estimations_at_one_time(hp, estimator, tt, nb_of_guesses, kernel_weight=kernel_plain, time_estimation=0,
                                  silent=False):
    for i in range(nb_of_guesses):
        if not silent:
            if i % 1 == 0:
                print(f"estimation {i} out of {nb_of_guesses} estimations.")
        else:
            if i % 20 == 0:
                print(f"estimation {i} out of {nb_of_guesses} estimations.")
        estimation_hp(hp, estimator, tt, kernel_weight=kernel_weight, time_estimation=time_estimation, silent=silent,
                      nb_of_guesses=nb_of_guesses)

    return # no need to return the estimator.























# those two functions are quite useless right now. I don't want to spend time on them to refractor them.





'''
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