##### normal libraries


##### my libraries



from classes.class_statistic_plot_estimator import *
from errors import Error_forbidden

##### other files
from classes.class_Estimator_Hawkes import *


# batch_estimation is one dataframe with the estimators.
class Statistic_plot_estimator_Hawkes(Statistic_plot_estimator):

    def __init__(self, estimator, fct_parameters):
        # BIANCA a way to have one constructor for all of them?
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator
        super().__init__(estimator, separators=['parameter', 'm', 'n'])

        # parameters is a list of lists of lists of functions
        self.ALPHA = fct_parameters[1]
        self.BETA = fct_parameters[2]  # makes the file more readable.
        self.NU = fct_parameters[0]
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(self.BETA))
        self.T_max = estimator.DF["T_max"].max()
        self.M = np.shape(self.ALPHA)[1]
        self.nb_of_guesses = estimator.DF['number of guesses'].max()

    @classmethod
    def from_path(cls, path, parameters):
        # BIANCA a way to have one constructor for all of them?
        # path has to be raw. with \\
        estimator = Estimator_Hawkes()
        estimator.append(pd.read_csv(path))
        return cls(estimator, parameters)

    def get_computation_plot_fig_dict(self, convergence_in):
        # todo the fig_dict could be more general to adapt to some situations, for now I simply put an if statement.
        if convergence_in == "MSE":
            fig_dict = {
                'title': f"Convergence in MSE sense of the estimators, batches of {self.nb_of_guesses} realisations.",
                'xlabel': "Nb of Events",
                'ylabel': "MSE",
                'parameters': [self.ALPHA[0][0](0, 1, 1), self.BETA[0][0](0, 1, 1), self.NU[0](0, 1, 1)],
                'name_parameters': ["ALPHA", "BETA", "NU"]
            }
            return fig_dict
        else:
            raise Error_forbidden

    def rescale_time_plot(self, mini_T, times):
        # I multiply by 50 bc I convert the time axis to jump axis, and a mini T corresponds to 50 jumps.
        return [times[i] // mini_T * 50 for i in range(len(times))]

    def rescale_sum(self, sum, times):
        '''
        rescale the data, for instance the MSE. The method is useful bc I can rescale with attributes.

        Args:
            sum:
            times:

        Returns:

        '''
        return sum / self.nb_of_guesses
