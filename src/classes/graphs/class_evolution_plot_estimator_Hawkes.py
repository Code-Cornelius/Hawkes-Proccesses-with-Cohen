##### normal libraries


##### my libraries


##### other files
from classes.graphs.class_evolution_plot_estimator import *

from classes.class_Estimator_Hawkes import Estimator_Hawkes
from classes.class_kernel import *


# section ######################################################################
#  #############################################################################
# code:

class Evolution_plot_estimator_Hawkes(Evolution_plot_estimator):
    evolution_name = 'time estimation'

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
        '''
        create another init that takes the same parameter, with the diff that it takes the path. Another constructor :

        Args:
            path:
            parameters:

        Returns:

        '''
        # BIANCA a way to have one constructor for all of them?
        # path has to be raw. with \\
        estimator = Estimator_Hawkes()
        estimator.append(pd.read_csv(path))
        return cls(estimator, parameters)

    # section ######################################################################
    #  #############################################################################
    # data

    @classmethod
    def get_evolution_name_unique_values(cls, data):
        return data[cls.evolution_name].unique()

    @classmethod
    def get_evolution_name_extremes(cls, data):
        values = data.groupby([cls.evolution_name])['value']
        return (values.min(), values.max())

    def get_evolution_name_true_value(self, data):
        return self.get_evolution_name_specific_data(data, 'true value')


    def get_evolution_name_plot_data(self, data):
        return self.get_evolution_name_specific_data(data, 'value')


    def get_evolution_name_specific_data(self, data, str):
        '''
        returns the data grouped by the particular attribute, and we focus on data given by column str, computing the means and returning an array.

        Args:
            data:
            str:

        Returns:

        '''
        return data.groupby([self.evolution_name])[str].mean().values

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_dict_fig(self, separators, key):
        title = self.generate_title(names=separators,
                                    values=key,
                                    before_text="",
                                    extra_text="Only 5-95% of the interval is shown, batches of {} simulations, time: 0 until {}",
                                    extra_arguments=[self.nb_of_guesses, self.T_max])
        fig_dict = {'title': "Evolution of the estimation, " + title,
                    'xlabel': 'Time',
                    'ylabel': "Estimation"}
        return fig_dict

    def draw(self, separators=None, separator_colour=None, kernel_plot_param=None,
             one_kernel_plot_param=None):
        '''
        plot the evolution of the estimators over the attribute given by get_plot_data.
        It is almost the same version as the upper class, the difference lies in that I m drawing the kernel on the graph additionally.
        I draw the kernels iff I give kernel_plot_param.

        kernel_plot_param for drawing over a list of kernels, one_kernel_plot for drawing the kernels in the middle.

        Args:
            separators:
            separator_colour: the column of the dataframe to consider for color discrimination
            kernel_plot_param:     list_of_kernels, Times = kernel_plot_param
            one_kernel_plot_param
        Returns:

        '''
        #we use the coloured keys for identifying which colors goes to whom in the one kernel plot case. We assume in the list_of_kernels all name are unique.
        _, coloured_keys = super().draw(separators, separator_colour)
        if kernel_plot_param is not None:
            list_of_kernels, Times = kernel_plot_param

            # here is all the plots I draw. I start at 1 bc I always plot the parameters as a first drawing.
            list_of_plots = APlot.print_register()[1:]
            # on each plot
            for counter, plots in enumerate(list_of_plots):
                # for each eval point
                for number, (kernel, a_time) in enumerate(zip(list_of_kernels, Times)):
                    if not number % (len(Times) // 14):  # I don't want to plot all the kernels, so only one upon 8 are drawn.
                        tt = [np.linspace(0, self.T_max, 3000)]
                        yy = kernel.eval(tt, a_time, self.T_max)
                        plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                              dict_plot_param={"color": "m", "markersize": 0, "linewidth": 0.4,
                                                               "linestyle": "--"}, tight=False)
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

                # we use the coloured keys for identifying which colors goes to whom in the one kernel plot case. We assume in the list_of_kernels all name are unique.
                for number, (kernel_name, color) in enumerate(zip(coloured_keys, colors)):
                    # basically, we retrieve the name and find the matching kernel.
                    kernel_counter = 0
                    kernel = None
                    while kernel is None and kernel_counter < len(list_of_kernels):
                        if list_of_kernels[kernel_counter].name == kernel_name:
                            kernel = list_of_kernels[kernel_counter]
                        else:
                            kernel_counter +=1
                    if kernel_counter > len(list_of_kernels):
                        raise("The kernels given and ploted are not matching.")
                    tt = [np.linspace(self.T_max * 0.05, self.T_max * 0.95, 3000)]
                    yy = kernel.eval(tt, Time, self.T_max)
                    plots.uni_plot_ax_bis(nb_ax=0, xx=tt[0], yy=yy[0],
                                          dict_plot_param={"color": color, "markersize": 0, "linewidth": 0.7,
                                                           "linestyle": "--"}, tight=False)
                # lim_ = plots.axs[0].get_ylim()
                # plots.plot_vertical_line(Time, np.linspace(0, lim_[-1] * 0.9, 5), nb_ax=0,
                #                         dict_plot_param={"color": "k", "markersize": 0, "linewidth": 1,
                #                         "linestyle": "--"})
                name_file = 'double_estimation_result_{}'.format(counter)
                plots.save_plot(name_save_file=name_file)
