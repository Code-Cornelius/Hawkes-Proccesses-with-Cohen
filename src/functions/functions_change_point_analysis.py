# https://github.com/amanahuja/change-detection-tutorial/blob/master/ipynb/section_tmp_all.ipynb
# https://github.com/deepcharles/ruptures


# normal libraries
import ruptures as rpt
import matplotlib.pyplot as plt

# other files
from classes.graphs.class_graph_estimator_hawkes import *
from classes.class_kernel import *


# my libraries
from library_errors.Error_not_yet_allowed import Error_not_yet_allowed
from library_errors.Error_not_enough_information import Error_not_enough_information


def change_point_analysis_and_plot(path=None, estimator_hawkes=None,
                                   type_analysis = "optimal",
                                   parameters_for_analysis = (1,"l2", 1),
                                   column_for_multi_plot_name=None):
    '''

    Args:
        path:  path is where the file is located where one can read the estimator Hawkes.
        estimator_hawkes:
        fct_parameters:
        width:
        min_size:
        number_of_breakpoints:
        model:
        column_for_multi_plot_name:

    Returns:

    '''
    # number of breakpoints doesn't support a different value of breakpoints for each variable.
    # path should be with \\
    #
    # column_for_multi_plot_name a string

    if type_analysis == "optimal":
        number_of_breakpoints, model, min_size = parameters_for_analysis

    elif type_analysis == "window":
        number_of_breakpoints, model, width = parameters_for_analysis
    else:
        raise Error_not_yet_allowed("Not good type of analysis.")



    if estimator_hawkes is None:
        the_estimator = Estimator_Hawkes.from_path(path)

    elif path is None:
        the_estimator = estimator_hawkes

    else:
        raise Error_not_enough_information("Path and Estimator_Hawkes can't be both None.")






    SEPARATORS = ['parameter', 'm', 'n']

    dict_serie = {}
    global_dict = the_estimator.DF.groupby(SEPARATORS)
    for k1, k2, k3 in global_dict.groups.keys():
        if column_for_multi_plot_name is not None:
            super_dict = global_dict.get_group((k1, k2, k3)).groupby([column_for_multi_plot_name])
            for k4 in super_dict.groups.keys():
                # discrimination of whether the serie already exists.
                if (k1, k2, k3) not in dict_serie:  # not yet crossed those values
                    dict_serie[(k1, k2, k3)] = super_dict.get_group(k4).groupby(['time estimation'])[
                        'value'].mean().values.reshape((1, -1))
                else:  # the condition already seen, so I aggregate to what was already done.
                    dict_serie[(k1, k2, k3)] = np.vstack((dict_serie[(k1, k2, k3)],
                                                          super_dict.get_group(k4).groupby(['time estimation'])[
                                                              'value'].mean()
                                                          ))
        else:
            dict_serie[(k1, k2, k3)] = global_dict.get_group((k1, k2, k3)).groupby(['time estimation'])[
                'value'].mean().values.reshape((1, -1))

    for k in dict_serie.keys():  # iterate through dictionary
        dict_serie[k] = np.transpose(dict_serie[k])

    ############################################## dynamic programming   http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/detection/dynp.html
    ans = []
    for k in dict_serie.keys():
        if type_analysis == "optimal":
            algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(dict_serie[k])
            my_bkps1 = algo.predict(n_bkps=number_of_breakpoints)
            rpt.show.display(dict_serie[k], my_bkps1, figsize=(10, 6))

        elif type_analysis == "window":
            algo = rpt.Window(width=width, model=model).fit(dict_serie[k])
            my_bkps1 = algo.predict(n_bkps=1)
            rpt.show.display(dict_serie[k], my_bkps1, figsize=(10, 6))
        ans.append(my_bkps1)
    return ans