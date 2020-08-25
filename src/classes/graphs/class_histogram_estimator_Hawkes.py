##### normal libraries


##### my libraries
from library_classes.graphs.class_histogram_estimator import *

##### other files
from classes.class_Estimator_Hawkes import *


# batch_estimation is one dataframe with the estimators.
from classes.graphs.class_Graph_Estimator_Hawkes import Graph_Estimator_Hawkes


class Histogram_estimator_Hawkes(Histogram_estimator, Graph_Estimator_Hawkes):
    nb_of_bins = 60

    def __init__(self, estimator, fct_parameters, *args, **kwargs):
        print( [x.__name__ for x in Histogram_estimator_Hawkes.__mro__] )

        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator
        #super().__init__(self, estimator = estimator, separators = ['parameter', 'm', 'n'], fct_parameters = fct_parameters, *args, **kwargs)
        Graph_Estimator_Hawkes.__init__(self, estimator, fct_parameters)
        Histogram_estimator.__init__(self, estimator, separators=['parameter', 'm', 'n'])


    # section ######################################################################
    #  #############################################################################
    # data

    # section ######################################################################
    #  #############################################################################
    # plot

    def get_range(self, key, mean):
        '''The best range for parameters is the following. It is then scaled up depending on the mean value.
        Args:
            key:
            mean:

        Returns:

        '''
        variable = key[0]
        if variable == "nu":
            return (0.1, 1.5 * mean)
        else:
            return (0.6 * mean, 1.4 * mean)

    # TODO: make more general -- don't assume that the name will always be the first
    def get_dict_param(self, key, mean):
        range = self.get_range(key, mean)
        dict_param = {'bins': Histogram_estimator_Hawkes.nb_of_bins,
                      'label': 'Histogram',
                      'color': 'green',
                      'range': range,
                      'cumulative': True
                      }
        return dict_param

    def get_dict_fig(self, separators, key):
        title = self.generate_title(names=separators, values=key,
                                    before_text="Histogram for the estimator of a Hawkes Process;",
                                    extra_text="Time of simulation {}", extra_arguments=[self.T_max])
        fig_dict = {'title': title,
                    'xlabel': "Estimation",
                    'ylabel': "Nb of realisation inside a bin."}
        return fig_dict
