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
        variable = key[0]
        if variable == "nu":
            return (0, 2 * mean)
        else:
            return (0.5 * mean, 1.5 * mean)

    # TODO: make more general -- don't assume that the name will always be the first
    def get_dict_param_for_plot(self, key, mean):
        range = self.get_optimal_range_histogram(key, mean)
        dict_param = {'bins': 30,
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

    def get_fig_dict_plot(self, separators, key):
        title = self.generate_title(separators, key, "",
                                    "Only 10-90% of the interval is shown (boundary effect), starting from 0 until {}.",
                                    [self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation" + title,
                    'xlabel': 'Time',
                    'ylabel': 'Value'}
        return fig_dict
    # BIANCA estimation in another file
    def get_extremes(self, data):
        estimation = data['time estimation'].unique()
        values = data.groupby(['time estimation'])['value']
        return (values.min(), values.max(), estimation)
    #### create another init that takes the same parameter, with the diff that it takes the path.
    # another constructor :
    def get_true_values(self, data):
        return self.get_specific_data(data, 'true value')

    def get_plot_data(self, data):
        return self.get_specific_data(data, 'value')

    # BIANCA 'time estimation' outside.
    def get_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute,
        and we focus on data given by column str, computing the means and returning an array.

        :param data:
        :param str:
        :return:
        '''
        return data.groupby(['time estimation'])[str].mean().values

    def get_computation_plot_fig_dict(self):
        fig_dict = {
            'title': "Convergence in compute_MSE of the estimators, batches of {} realisations.".format(self.nb_of_guesses),
            'labels': ["Nb of Events", "compute_MSE of the Estimator"],
            'parameters': [self.ALPHA[0][0](0, 1), self.BETA[0][0](0, 1), self.NU[0](0, 1)],
            'name_parameters':["ALPHA", "BETA", "NU"]
        }
        return fig_dict

    def get_times_plot(self, mini_T, times):
        return [times[i] // mini_T * 50 for i in range(len(times))]

    def rescale_sum(self, sum, times):
        return sum / self.nb_of_guesses
