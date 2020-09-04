# normal libraries
import warnings

# my libraries
from library_errors.Error_convergence import *

# other files
from classes.class_hawkes_process import *
from functions.functions_derivatives_MLE import *
from functions.functions_my_newton_raphson import newtons_method_multi_MLE


# the function returns a flag for the reason beeing that if it failed to converge too many times,
# it s perhaps better to try on a new data set.
# now there is an error raised.
def simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation):
    T_max = tt[-1]
    _, time_real = hp.simulation_Hawkes_exact_with_burn_in(tt=tt, plot_bool=False,
                                                           silent=True)  # don't store intensity, only used for plots.
    w = kernel_weight.eval(T_t=time_real, eval_point=time_estimation, T_max=T_max)

    # One shouldn't get an infinite loop here, at some point, the algorithm should converge.
    # There is also a warning triggered if algorithm didn't converged.
    try:
        alpha_hat, beta_hat, mu_hat = call_newton_raph_MLE_opt(time_real, T_max, w, silent=silent)
    except Error_convergence as err:
        warnings.warn(err.message)
        return simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation)
    return alpha_hat, beta_hat, mu_hat


def call_newton_raph_MLE_opt(T_t, T, w=None, silent=True):
    # w shouldn't be None, however as a safety measure, just before doing the computations !
    if w is None:
        w = Kernel(fct_plain, "plain", T_max=T).eval(T_t, 0, T_max=T)
        # eval point equals 0 because, if the weights haven't been defined earlier,
        # it means we don't care when we estimate.
    M = len(T_t)
    NU = np.full(M, 0.1)
    ALPHA = np.full((M, M), 0.7)
    BETA = 0.2 + 1.1 * M * M * ALPHA

    # ALPHA = np.array([[2, 1], [1, 2]]) * 0.99
    # BETA = np.array([[5, 3], [3, 5]]) * 0.99
    # NU = np.array([0.2, 0.2]) * 0.99

    # ALPHA = np.array([[1, 2], [1, 2]])
    # BETA = np.array([[5, 10], [5, 10]])
    # NU = np.array([1, 1])

    # print("debug")
    # T = 20
    # print(T_t)
    # T_t = [[17, 18, 19], [16, 19]]
    # w = Kernel(fct_plain, "plain").eval(T_t, 0)
    # print("debug")

    df = lambda nu, alpha, beta: first_derivative(T_t, alpha, beta, nu, T, w)
    ddf = lambda nu, alpha, beta: second_derivative(T_t, alpha, beta, nu, T, w)

    NU, ALPHA, BETA = newtons_method_multi_MLE(df, ddf, ALPHA, BETA, NU, silent=silent)
    return ALPHA, BETA, NU


def estimation_hp(hp, estimator, tt, nb_of_guesses, kernel_weight=kernel_plain, time_estimation=0,
                  silent=True):
    """

    Args:
        hp:  hawkes process
        estimator:
        tt:
        nb_of_guesses:
        kernel_weight:  should be ONE kernel from class_kernel.
        time_estimation:
        silent:

    Returns:

    """

    alpha_hat, beta_hat, mu_hat = simulation_and_convergence(tt, hp, kernel_weight, silent, time_estimation)

    _, M = np.shape(alpha_hat)
    T_max = tt[-1]
    for s in range(M):
        estimator.DF = estimator.DF.append(pd.DataFrame(
            {"time estimation": [time_estimation],
             "parameter": ["nu"],
             "n": [s],
             "m": [0],
             "weight function": [kernel_weight.name],
             "value": [mu_hat[s]],
             'T_max': [T_max],
             'time_burn_in': [Hawkes_process.TIME_BURN_IN],
             'true value': [hp.NU[s](time_estimation, T_max, Hawkes_process.TIME_BURN_IN)],
             'number of guesses': [nb_of_guesses]
             }), sort=True
        )
        for t in range(M):
            estimator.DF = estimator.DF.append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "parameter": ["alpha"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [alpha_hat[s, t]],
                 'T_max': [T_max],
                 'time_burn_in': [Hawkes_process.TIME_BURN_IN],
                 'true value': [hp.ALPHA[s][t](time_estimation, T_max, Hawkes_process.TIME_BURN_IN)],
                 'number of guesses': [nb_of_guesses]
                 }), sort=True
            )
            estimator.DF = estimator.DF.append(pd.DataFrame(
                {"time estimation": [time_estimation],
                 "parameter": ["beta"],
                 "n": [s],
                 "m": [t],
                 "weight function": [kernel_weight.name],
                 "value": [beta_hat[s, t]],
                 'T_max': [T_max],
                 'time_burn_in': [Hawkes_process.TIME_BURN_IN],
                 'true value': [hp.BETA[s][t](time_estimation, T_max, Hawkes_process.TIME_BURN_IN)],
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

    return  # no need to return the estimator.