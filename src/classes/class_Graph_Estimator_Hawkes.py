##### normal libraries


##### my libraries

##### other files
from classes.class_Estimator_Hawkes import *
from classes.class_kernel import *


# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    def __init__(self, estimator, fct_parameters):
        # Initialise the Graph with the estimator
        Graph_Estimator.__init__(self, estimator, ['variable', 'm', 'n'])

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
        return cls(estimator, parameters)

    #############################" hist

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

    ################################ hist

    def get_dict_fig_evolution_parameter_over_time(self, separators, key):
        title = self.generate_title(separators, key, "",
                                    "Only 5-95% of the interval is shown, batches of {} simulations, time: 0 until {}.",
                                    [self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation" + title,
                    'xlabel': 'Time',
                    'ylabel': 'Value'}
        return fig_dict

    evolution_name = 'time estimation'

    def get_evolution_parameter(self, data):
        return data[Graph_Estimator_Hawkes.evolution_name].unique()

    def get_evolution_extremes(self, data):
        values = data.groupby([Graph_Estimator_Hawkes.evolution_name])['value']
        return (values.min(), values.max())

    def get_evolution_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute,
        and we focus on data given by column str, computing the means and returning an array.

        :param data:
        :param str:
        :return:
        '''
        return data.groupby([Graph_Estimator_Hawkes.evolution_name])[str].mean().values

    #### create another init that takes the same parameter, with the diff that it takes the path.
    # another constructor :
    def get_evolution_true_value(self, data):
        return self.get_evolution_specific_data(data, 'true value')

    def get_evolution_plot_data(self, data):
        return self.get_evolution_specific_data(data, 'value')

    def get_computation_plot_fig_dict(self):
        fig_dict = {
            'title': f"Convergence in compute_MSE of the estimators, batches of {self.nb_of_guesses} realisations.",
            'labels': ["Nb of Events", "compute_MSE of the Estimator"],
            'parameters': [self.ALPHA[0][0](0, 1), self.BETA[0][0](0, 1), self.NU[0](0, 1)],
            'name_parameters': ["ALPHA", "BETA", "NU"]
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

    def draw_evolution_parameter_over_time(self, separators=None, separator_colour=None, plot_param=None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.
        It is almost the same version as the upper class, the difference lies in that I m drawing the kernel on the graph additionally.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination

        Returns:

        '''
        super().draw_evolution_parameter_over_time(separators, separator_colour)
        if plot_param is not None:
            list_of_kernels, Times = plot_param

            list_of_plots = APlot.print_register()
            # on each plot
            for counter, plots in enumerate(list_of_plots):
                # for each eval point
                for number, (kernel, a_time) in enumerate(zip(list_of_kernels, Times)):
                    if not number % 6: # I don't want to plot all the kernels, so only one upon 3 are drawn.
                        tt = [np.linspace(0, self.T_max, 10000)]
                        yy = kernel.eval(tt, a_time, self.T_max)
                        plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                              dict_plot_param={"color": "m", "markersize": 0, "linewidth": 0.4,
                                                               "linestyle": "--"})
                        lim_ = plots.axs[0].get_ylim()
                        plots.plot_vertical_line(a_time, np.linspace(0, lim_[-1] * 0.9, 5), nb_ax=0,
                                                 dict_plot_param={"color": "k", "markersize": 0, "linewidth": 0.2,
                                                                  "linestyle": "--"})
                name_file = 'double_estimation_result_{}'.format(counter)
                plots.save_plot(name_save_file=name_file)
