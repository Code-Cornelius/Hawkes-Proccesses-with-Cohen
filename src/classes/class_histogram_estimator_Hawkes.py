##### normal libraries


##### my libraries
from classes.class_Estimator_Hawkes import *
from classes.class_kernel import *
from errors import Error_forbidden
from classes.class_histogram_estimator import *


##### other files


# batch_estimation is one dataframe with the estimators.
class Histogram_estimator_Hawkes(Histogram_estimator):

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

    def get_optimal_range_histogram(self, key, mean):
        '''
        by experience, the best range for parameters is the following.
        It is then scaled up depending on the mean value.

        :param key:
        :param mean:
        :return:
        '''
        variable = key[0]
        if variable == "nu":
            return (0.1, 2 * mean)
        else:
            return (0.5 * mean, 1.5 * mean)

    # TODO: make more general -- don't assume that the name will always be the first
    def get_dict_plot_param_for_hist(self, key, mean):
        range = self.get_optimal_range_histogram(key, mean)
        dict_param = {'bins': 60,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig_hist(self, separators, key):
        title = self.generate_title(names=separators, values=key,
                                    before_text="Histogram for the estimator of a Hawkes Process;",
                                    extra_text="Time of simulation {}", extra_arguments=[self.T_max])
        fig_dict = {'title': title,
                    'xlabel': "Estimation",
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict
