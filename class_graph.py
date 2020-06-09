import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import plot_functions

from class_estimator import *
import functions_general_for_Hawkes

# batch_estimation is one dataframe with the estimators.
class Graph:
    def __init__(self, estimator, ALPHA, BETA, MU, T_max, nb_of_guesses):
        self.estimator = estimator
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.MU = MU
        self.T_max = T_max
        self.M = np.shape(self.ALPHA)[1]
        self.nb_of_guesses = nb_of_guesses

    #### create another init that takes the same parameter, with the diff that it takes the path.
    #another constructor :
    @classmethod
    def from_path(cls, path, nb_of_guesses, parameters):
        # path has to be raw. with \\
        estimator = Estimator(pd.read_csv(path))
        # get the max value which is M-1
        T_max = estimator.DF["T_max"].max()
        return cls(estimator, parameters[0], parameters[1], parameters[2], T_max, nb_of_guesses)


    def multi_simul_Hawkes_and_estimation(self):
        # print(list_of_succesive_coefficients)
        # plot the histograms for every coefficient
        # ecart correspond à true value * % de la valeur ( qui est range coef )


        # isolating estimation of each type
        #TODO ADAPT FOR MULTI DIMENSIONAL CASE
        estim_alpha = self.estimator.DF[self.estimator.DF['variable'] == "alpha"].copy()
        estim_beta = self.estimator.DF[self.estimator.DF['variable'] == "beta"].copy()
        estim_mu = self.estimator.DF[self.estimator.DF['variable'] == "nu"].copy()

        list_of_succesive_coefficients = np.zeros((self.nb_of_guesses, self.M + 2 * self.M * self.M)) # each line is one guess. So it's like "simul i ; x0"
        for i in range(self.nb_of_guesses):
            for j in range(self.M):
                list_of_succesive_coefficients[i][j] = estim_mu.iloc[i]["value"]
            for j in range(self.M * self.M):
                list_of_succesive_coefficients[i][j + self.M] = estim_alpha.iloc[i]["value"]
                list_of_succesive_coefficients[i][j + self.M + self.M * self.M] = estim_beta.iloc[i]["value"]


        for i in range(len(list_of_succesive_coefficients[0, :])):
            if i < self.M:
                title = " Histogram, estimator (time = {}) of the parameter NU, {} tries, {}th value, true value = {}.".format(
                    self.T_max,
                    self.nb_of_guesses, i + 1, self.MU[i])
                range_coef = (0, 2)
                ecart_plot = tuple(inner * self.MU[i] for inner in range_coef)
            elif i < self.M + self.M * self.M:
                title = " Histogram, estimator (time = {}) of the parameter ALPHA, {} tries, {}th value, true value = {}.".format(
                    self.T_max,
                    self.nb_of_guesses, i + 1 - self.M, np.ravel(self.ALPHA)[i - self.M])
                range_coef = (0.5, 1.5)
                ecart_plot = tuple(inner * np.ravel(self.ALPHA)[i - self.M] for inner in range_coef)
            else:
                title = " Histogram, estimator (time = {}) of the parameter BETA, {} tries, {}th value, true value = {}.".format(
                    self.T_max,
                    self.nb_of_guesses, i + 1 - self.M - self.M * self.M, np.ravel(self.BETA)[i - self.M * self.M - self.M])
                range_coef = (0.5, 1.5)
                ecart_plot = tuple(inner * np.ravel(self.BETA)[i - self.M - self.M * self.M] for inner in range_coef)

            plot_functions.hist(list_of_succesive_coefficients[:, i], 30, title, "value", range=ecart_plot,
                                total_number_of_simulations=self.nb_of_guesses)
        return

    # function estimation over time:
    def estimation_hawkes_parameter_over_time(self, list_of_func, **kwargs):
        # the list of func allows me to plot the line of true parameters.
        # the kwargs are for the functions.

        # isolating estimation of each type
        #TODO ADAPT FOR MULTI DIMENSIONAL CASE
        estim_alpha = self.estimator.DF[self.estimator.DF['variable'] == "alpha"].copy()
        estim_alpha_extr = (estim_alpha.groupby(['time estimation'])['value'].min(),
                            estim_alpha.groupby(['time estimation'])['value'].max())

        estim_beta = self.estimator.DF[self.estimator.DF['variable'] == "beta"].copy()
        estim_beta_extr = (estim_beta.groupby(['time estimation'])['value'].min(),
                           estim_beta.groupby(['time estimation'])['value'].max())

        estim_mu = self.estimator.DF[self.estimator.DF['variable'] == "nu"].copy()
        estim_mu_extr = (estim_mu.groupby(['time estimation'])['value'].min() ,
                         estim_mu.groupby(['time estimation'])['value'].max() )

        extrem_time_estimation = estim_alpha['time estimation'].unique() # list of times where I have data to plot.

        plot = sns.relplot(data=estim_alpha, x="time estimation", y="value",
                           hue='weight function',  # style="weight function",
                           # markers=True, dashes=False,
                           kind="line", sort=True, ci=None)
        title = " Evolution of the estimation of the estimator of the parameter $\\alpha$, \
\nused {} estimations of the parameter. \
\nOnly 10-90% of the interval is shown (boundary effect), starting from 0 until {}.".\
            format(self.nb_of_guesses, self.T_max)
        plot.fig.suptitle(title, verticalalignment='top', fontsize=12)
        sous_text = " True Parameters : \n \
            ALPHA {}, \n     BETA {}, \n     NU {}".format(self.ALPHA, self.BETA, self.MU)
        plot.fig.text(0, 0.1, sous_text, fontsize=10)
        plt.plot(extrem_time_estimation, estim_alpha_extr[0],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        plt.plot(extrem_time_estimation, estim_alpha_extr[1],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        # TODO EXTREM CAREFUL TO LIST_OF_FUNC IN MULTI DIMENSIONAL
        plt.plot(extrem_time_estimation,
                 [ (list_of_func[0][0][0])(time, **kwargs) for time in extrem_time_estimation ],
                 linestyle = "solid", linewidth=0.4, color = "r")
        plt.savefig('estimation_alpha.png', dpi=800)





        plot = sns.relplot(data=estim_beta, x="time estimation", y="value",
                           hue='weight function',  # style="weight function",
                           # markers=True, dashes=False,
                           kind="line", sort=True, ci=None)
        title = " Evolution of the estimation of the estimator of the parameter $\\beta$, \
\nused {} estimations of the parameter. \
\nOnly 10-90% of the interval is shown (boundary effect), starting from 0 until {}.".\
            format(self.nb_of_guesses, self.T_max)
        plot.fig.suptitle(title, verticalalignment='top', fontsize=12)
        sous_text = " True Parameters : \n \
            ALPHA {}, \n     BETA {}, \n     NU {}".format(self.ALPHA, self.BETA, self.MU)
        plot.fig.text(0, 0.1, sous_text, fontsize=10)
        plt.plot(extrem_time_estimation, estim_beta_extr[0],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        plt.plot(extrem_time_estimation, estim_beta_extr[1],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        # TODO EXTREM CAREFUL TO LIST_OF_FUNC IN MULTI DIMENSIONAL
        plt.plot(extrem_time_estimation,
                 [ (list_of_func[1][0][0])(time, **kwargs) for time in extrem_time_estimation ],
                 linestyle = "solid", linewidth=0.4, color = "r")
        plt.savefig('estimation_beta.png', dpi=800)





        plot = sns.relplot(data=estim_mu, x="time estimation", y="value",
                           hue='weight function',  # style="weight function",
                           # markers=True, dashes=False,
                           kind="line", sort=True, ci=None)
        title = " Evolution of the estimation of the estimator of the parameter $\\nu$, \
\nused {} estimations of the parameter. \
\nOnly 10-90% of the interval is shown (boundary effect), starting from 0 until {}.".\
            format(self.nb_of_guesses, self.T_max)
        plot.fig.suptitle(title, verticalalignment='top', fontsize=12)
        sous_text = " True Parameters : \n \
            ALPHA {}, \n     BETA {}, \n     NU {}".format(self.ALPHA, self.BETA, self.MU)
        plot.fig.text(0, 0.1, sous_text, fontsize=10)
        plt.plot(extrem_time_estimation, estim_mu_extr[0],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        plt.plot(extrem_time_estimation, estim_mu_extr[1],
                 linestyle = "dashdot", linewidth=0.5, color = "r")
        # TODO EXTREM CAREFUL TO LIST_OF_FUNC IN MULTI DIMENSIONAL
        plt.plot(extrem_time_estimation,
                 [ (list_of_func[2][0][0])(time, **kwargs) for time in extrem_time_estimation ],
                 linestyle = "solid", linewidth=0.4, color = "r")
        plt.savefig('estimation_nu.png', dpi=800)
        return

    # TODO NOT DONE FUNCTION YET
    # for that function, compare value between result and true value for all the times, and plot.
    # i can compute with 2 for loops like before.
    def multi_simul_Hawkes_and_estimation_MSE(self, mini_T):
        def MSE(param, true_parameter):
            return (param - true_parameter) ** 2


        estim_alpha = Estimator(self.estimator.DF.loc[self.estimator.DF['variable'] == "alpha"].copy())
        estim_beta = Estimator(self.estimator.DF.loc[self.estimator.DF['variable'] == "beta"].copy())
        estim_mu = Estimator(self.estimator.DF.loc[self.estimator.DF['variable'] == "nu"].copy())

        #test si MEAN est ok
        if  estim_alpha.DF["true value"].nunique() != 1 or \
            estim_beta.DF["true value"].nunique() != 1 or \
            estim_mu.DF["true value"].nunique() != 1 :
            raise("Check something dude.")
        estim_alpha.function_upon_separeted_data("value", MSE, "MSE", true_parameter = estim_alpha.DF["true value"].mean() )
        estim_beta.function_upon_separeted_data("value", MSE, "MSE", true_parameter = estim_beta.DF["true value"].mean())
        estim_mu.function_upon_separeted_data("value", MSE, "MSE", true_parameter = estim_mu.DF["true value"].mean())

        DF_MSE = estim_mu.DF.groupby(['T_max'])["MSE"].sum()

        MSE_reals = np.zeros(len(self.T_max))
        TIMES_plot = [self.T_max[i] // mini_T * 50 for i in range(len(self.T_max))]
        i = 0
        for times in self.T_max:
            MSE_reals[i] = DF_MSE[times] / self.nb_of_guesses
            i += 1

        plot_functions.plot_graph(TIMES_plot, MSE_reals, title=[
            "Convergence in MSE of the estimators, batches of {} realisations.".format(self.nb_of_guesses)],
                                  labels=["Nb of Events", "MSE of the Estimator"],
                                  parameters=[self.ALPHA, self.BETA, self.MU], name_parameters=["ALPHA", "BETA", "NU"],
                                  name_save_file="MSE_convergence")

        #TODO I NEED THE HISTOGRAM OF LAST VALUE IN MSE...
        return







































