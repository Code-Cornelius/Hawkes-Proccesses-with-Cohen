##### normal libraries
##### my libraries

# errors:

##### other files
from classes.class_estimator_hawkes import *
from classes.class_kernel import *


# batch_estimation is one dataframe with the estimators.
class Graph_Estimator_Hawkes(Graph_Estimator):
    evolution_name = 'time estimation'
    separators = ['parameter', 'm', 'n']

    def __init__(self, estimator, fct_parameters, *args, **kwargs):
        # TODO IF FCT_PARAMETERS IS NONE, NOT PLOT TRUE VALUE, PERHAPS IT IS NOT KWOWN.
        # Initialise the Graph with the estimator

        super().__init__(estimator=estimator, fct_parameters=fct_parameters,
                         separators=Graph_Estimator_Hawkes.separators,
                         *args, **kwargs)

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
        return cls(estimator=estimator, fct_parameters=parameters)
