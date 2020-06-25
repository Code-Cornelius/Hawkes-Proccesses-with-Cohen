##### normal libraries


##### my libraries

##### other files
from classes.class_Estimator_Hawkes import *
from classes.class_kernel import *


# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    def __init__(self, estimator, fct_parameters):
        # Initialise the Graph with the estimator
        super().__init__(estimator, ['variable', 'm', 'n'])

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
        # path has to be raw. with \\
        estimator = Estimator_Hawkes()
        estimator.append(pd.read_csv(path))
        # get the max value which is M-1
        return cls(estimator, parameters)

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
        dict_param = {'bins': 35,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig_hist(self, separators, key):
        title = self.generate_title(separators, key)
        fig_dict = {'title': "Histogram" + title,
                    'xlabel': 'value',
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict

    def get_dict_fig_evolution_parameter_over_time(self, separators, key):
        title = self.generate_title(separators, key, "",
                                    "Only 10-90% of the interval is shown (boundary effect), starting from 0 until {}.",
                                    [self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation" + title,
                    'xlabel': 'Time',
                    'ylabel': 'Value'}
        return fig_dict

    evolution_name = 'time estimation'
    def get_evolution_parameter(self,data):
        return data[Graph_Estimator_Hawkes.evolution_name].unique()

    def get_evolution_extremes(self, data):
        values = data.groupby([Graph_Estimator_Hawkes.evolution_name])['value']
        return (values.min(), values.max())
    #### create another init that takes the same parameter, with the diff that it takes the path.
    # another constructor :
    def get_evolution_true_value(self, data):
        return self.get_evolution_specific_data(data, 'true value')

    def get_evolution_plot_data(self, data):
        return self.get_evolution_specific_data(data, 'value')

    def get_evolution_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute,
        and we focus on data given by column str, computing the means and returning an array.

        :param data:
        :param str:
        :return:
        '''
        return data.groupby([Graph_Estimator_Hawkes.evolution_name])[str].mean().values

    def get_computation_plot_fig_dict(self):
        fig_dict = {
            'title': "Convergence in compute_MSE of the estimators, batches of {} realisations.".format(self.nb_of_guesses),
            'labels': ["Nb of Events", "compute_MSE of the Estimator"],
            'parameters': [self.ALPHA[0][0](0, 1), self.BETA[0][0](0, 1), self.NU[0](0, 1)],
            'name_parameters':["ALPHA", "BETA", "NU"]
        }
        return fig_dict

    def rescale_time_plot(self, rescale_factor, times):
        # I multiply by 50 bc I convert the time axis to jump axis, and a mini T corresponds to 50 jumps.
        return [times[i] // rescale_factor * 50 for i in range(len(times))]

    def rescale_sum(self, sum, times):
        '''
        rescale the data, for instance the MSE. The method is useful bc I can rescale with attributes.

        :param sum:
        :param times:
        :return:
        '''
        return sum / self.nb_of_guesses
