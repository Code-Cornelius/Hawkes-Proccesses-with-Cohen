##### normal libraries


##### my libraries
from classes.class_Estimator_Hawkes import *
from classes.class_kernel import *
from errors import Error_forbidden

##### other files



# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    evolution_name = 'time estimation'

    def __init__(self, estimator, fct_parameters):

        #TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator
        Graph_Estimator.__init__(self, estimator, ['parameter', 'm', 'n'])

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
        dict_param = {'bins': 60,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig_hist(self, separators, key):
        title = self.generate_title(names = separators, values = key, before_text = "Histogram for the estimator of a Hawkes Process;",
                                    extra_text="Time of simulation {}", extra_arguments=[self.T_max])
        fig_dict = {'title': title,
                    'xlabel': "Estimation",
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict


    ################################ hist

    def get_dict_fig_evolution_parameter_over_time(self, separators, key):
        title = self.generate_title(names = separators,
                                    values = key,
                                    before_text = "",
                                    extra_text = "Only 5-95% of the interval is shown, batches of {} simulations, time: 0 until {}",
                                    extra_arguments = [self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation, " + title,
                    'xlabel': 'Time',
                    'ylabel': "Estimation"}
        return fig_dict


    @staticmethod
    def get_evolution_parameter(data):
        return data[Graph_Estimator_Hawkes.evolution_name].unique()

    @staticmethod
    def get_evolution_extremes(data):
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

    def get_computation_plot_fig_dict(self, convergence_in):
        # todo the fig_dict could be more general to adapt to some situations, for now I simply put an if statement.
        if convergence_in == "MSE":
            fig_dict = {
                'title': f"Convergence in MSE sense of the estimators, batches of {self.nb_of_guesses} realisations.",
                'xlabel': "Nb of Events",
                'ylabel':"MSE",
                'parameters': [self.ALPHA[0][0](0, 1, 1), self.BETA[0][0](0, 1,1), self.NU[0](0, 1,1)],
                'name_parameters': ["ALPHA", "BETA", "NU"]
            }
            return fig_dict
        else : raise Error_forbidden

    def rescale_time_plot(self, rescale_factor, times):
        # I multiply by 50 bc I convert the time axis to jump axis, and a mini T corresponds to 50 jumps.
        return [times[i] // rescale_factor * 50 for i in range(len(times))]

    def rescale_sum(self, sum, times):
        '''
        rescale the data, for instance the MSE. The method is useful bc I can rescale with attributes.

        Args:
            sum:
            times:

        Returns:

        '''
        return sum / self.nb_of_guesses

    def draw_evolution_parameter_over_time(self, separators=None, separator_colour=None, kernel_plot_param=None, one_kernel_plot_param = None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.
        It is almost the same version as the upper class, the difference lies in that I m drawing the kernel on the graph additionally.
        I draw the kernels iff I give kernel_plot_param.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination
            kernel_plot_param:     list_of_kernels, Times = kernel_plot_param
        Returns:

        '''
        super().draw_evolution_parameter_over_time(separators, separator_colour)
        if kernel_plot_param is not None:
            list_of_kernels, Times = kernel_plot_param

            # here is all the plots I draw. I start at 1 bc I always plot the parameters as a first drawing.
            list_of_plots = APlot.print_register()[1:]
            # on each plot
            for counter, plots in enumerate(list_of_plots):
                # for each eval point
                for number, (kernel, a_time) in enumerate(zip(list_of_kernels, Times)):
                    if not number % (len(Times)//3): # I don't want to plot all the kernels, so only one upon 3 are drawn.
                        tt = [np.linspace(0, self.T_max, 3000)]
                        yy = kernel.eval(tt, a_time, self.T_max)
                        plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                              dict_plot_param={"color": "m", "markersize": 0, "linewidth": 0.4,
                                                               "linestyle": "--"}, tight = False)
                        lim_ = plots.axs[0].get_ylim()
                        plots.plot_vertical_line(a_time, np.linspace(0, lim_[-1] * 0.9, 5), nb_ax=0,
                                                 dict_plot_param={"color": "k", "markersize": 0, "linewidth": 0.2,
                                                                  "linestyle": "--"})
                name_file = 'double_estimation_result_{}'.format(counter)
                plots.save_plot(name_save_file=name_file)

        elif one_kernel_plot_param is not None:
            list_of_kernels, Time = one_kernel_plot_param

            # here is all the plots I draw. I start at 1 bc I always plot the parameters as a first drawing.
            list_of_plots = APlot.print_register()[1:]
            # on each plot
            for counter, plots in enumerate(list_of_plots):
                # for each eval point

                colors = plt.cm.Dark2.colors  # Dark2 is qualitative cm and pretty dark cool colors.
                for number, (kernel,color) in enumerate(zip(list_of_kernels, colors)):
                        tt = [np.linspace(self.T_max * 0.05, self.T_max * 0.95, 3000)]
                        yy = kernel.eval(tt, Time, self.T_max)
                        plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                              dict_plot_param={"color": color, "markersize": 0, "linewidth": 0.7,
                                                               "linestyle": "--"}, tight = False)
                # lim_ = plots.axs[0].get_ylim()
                # plots.plot_vertical_line(Time, np.linspace(0, lim_[-1] * 0.9, 5), nb_ax=0,
                #                         dict_plot_param={"color": "k", "markersize": 0, "linewidth": 1,
                #                         "linestyle": "--"})
                name_file = 'double_estimation_result_{}'.format(counter)
                plots.save_plot(name_save_file=name_file)