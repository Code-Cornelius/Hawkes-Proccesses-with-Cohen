# normal libraries
import warnings

##### my libraries
from errors.error_convergence import *

##### other files
from functions_MLE_algo import newtons_method_multi_MLE
from functions_derivatives_MLE import *
from classes.class_hawkes_process import *


# the function returns a flag for the reason beeing that if it failed to converge too many times, it s perhaps better to try on a new data set.
def call_newton_raph_MLE_opt(T_t, T, w=None, silent=True):
    if w is None:
        w = Kernel(fct_plain, "plain").eval(T_t,
                                            0)  # eval point equals 0 because, if the weights haven't been defined earlier, it means we don't care when we estimate.

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

def estimation_hp(hp, estimator, T_max, nb_of_guesses, kernel_weight= kernel_plain, time_estimation=0,
                  silent=True):
    ## function_weight should be ONE kernel from class_kernel.
    ## hp is a hawkes process
    ## the flag notes if the convergence was a success. If yes, function hands in the results

    alpha_hat, beta_hat, mu_hat = simulation_and_convergence(T_max, hp, kernel_weight, silent, time_estimation)

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
             'number of guesses': [nb_of_guesses]
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
                 'number of guesses': [nb_of_guesses]
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
                 'number of guesses': [nb_of_guesses]
                 }), sort=True
            )
    return  # no need to return the estimator.


def simulation_and_convergence(T_max, hp, kernel_weight, silent, time_estimation):

    intensity, time_real = hp.simulation_Hawkes_exact(T_max=T_max, plot_bool=False, silent=True)
    print(time_real)
    w = kernel_weight.eval(T_t=time_real, eval_point=time_estimation)
    try:
        alpha_hat, beta_hat, mu_hat = call_newton_raph_MLE_opt(time_real, T_max, w, silent=silent)
    except Error_convergence as err:
        warnings.warn(err.message)
        return simulation_and_convergence(T_max, hp, kernel_weight, silent, time_estimation)
    return alpha_hat, beta_hat, mu_hat


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