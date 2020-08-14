# normal libraries
import unittest
import recurrent_functions
import matplotlib.pyplot as plt
# my libraries
import plot_functions
# other files
from functions_fct_evol_parameters import *

np.random.seed(124)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Test_images(unittest.TestCase):

    def tearDown(self):
        plt.show()

    def test_image_FKDE_AKDE_CKDE(self):
        nb_of_points = 10000
        ############################## 1
        xx = np.linspace(-15, 15, nb_of_points)
        mesh = 30. / nb_of_points
        zz = np.zeros(nb_of_points)
        my_plot = plot_functions.APlot(how=(1, 1))
        points = np.array([-7., -6., 1., 2., 5.])
        for f in points:
            yy = recurrent_functions.phi_numpy(xx, f, 2) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'})
            zz += yy

        print(np.sum(zz * mesh))

        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'})
        my_plot.set_dict_fig(0, {'xlabel': 'Time $t$', 'ylabel': 'Probability',
                                 'title': 'KDE estimation, fixed size kernel'})
        my_plot.show_legend()

        ############################## 2
        zz = np.zeros(nb_of_points)
        my_plot = plot_functions.APlot(how=(1, 1))
        points = np.array([-7., -6., 1., 2., 5.])
        for f in points:
            yy = recurrent_functions.phi_numpy(xx, f, 2 * (1 + math.fabs(f) / 10)) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'})
            zz += yy

        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'})
        my_plot.set_dict_fig(0,
                             {'xlabel': 'Time $t$', 'ylabel': 'Probability',
                              'title': 'KDE estimation, adaptive size kernel'})
        my_plot.show_legend()

        print(np.sum(zz * mesh))

        ############################## 3
        ############### left
        zz = np.zeros(nb_of_points)
        max_x = 0.125
        my_plot = plot_functions.APlot(how=(1, 2), sharey=True)
        my_plot.uni_plot(0, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        points = np.array([-1.1, 0.5, 5.])
        for f in points:
            my_plot.uni_plot(0, [f for _ in xx],
                             np.linspace(-0.004, max_x, len(xx)),
                             dict_plot_param={'color': 'k', 'linestyle': '--', 'linewidth': 0.7,
                                              'markersize': 0,
                                              'label': None})
            yy = recurrent_functions.phi_numpy(xx, f, 2) / len(points)
            my_plot.uni_plot(0, xx, yy, dict_plot_param={'label': f'Kernel at {f}'})
            my_plot.uni_plot(0, xx[nb_of_points // 2], yy[nb_of_points // 2],
                             dict_plot_param={'color': 'r',
                                              'markersize': 8, 'marker': '*', 'label': None})
            zz += yy
        my_plot.uni_plot(0, xx, zz, dict_plot_param={'color': 'r', 'label': 'KDE'})
        print("value : ", zz[nb_of_points // 2])

        ############### right
        zz = recurrent_functions.phi_numpy(xx, 0, 2) / 3
        print(np.sum(zz * mesh))
        for f in points:
            my_plot.uni_plot(1, [f for _ in xx],
                             np.linspace(-0.004, recurrent_functions.phi_numpy(f, 0, 2) / 3, len(xx)),
                             dict_plot_param={'color': 'm', 'linestyle': '--', 'linewidth': 0.7,
                                              'markersize': 0,
                                              'label': f'Value kernel at {f}'})
            my_plot.uni_plot(1, f, recurrent_functions.phi_numpy(f, 0, 2) / 3,
                             dict_plot_param={'color': 'g',
                                              'markersize': 8, 'marker': '*', 'label': None})

        my_plot.uni_plot(1, xx, zz, dict_plot_param={'color': 'r', 'label': 'Kernel for $t = 0$'})

        ### sum
        previous_f = 0

        for i, (f, c) in enumerate(zip(points, ['b', 'c', 'k'])):
            my_plot.uni_plot(1, -15, previous_f + recurrent_functions.phi_numpy(f, 0, 2) / 3,
                             dict_plot_param={'color': c,
                                              'markersize': 8, 'marker': '*',
                                              'label': f'cumsum leading to true result {i}'})
            my_plot.uni_plot(1, [-15 for _ in xx],
                             np.linspace(previous_f, previous_f + recurrent_functions.phi_numpy(f, 0, 2) / 3, len(xx)),
                             dict_plot_param={'color': c, 'linestyle': '--', 'linewidth': 0.7,
                                              'markersize': 0,
                                              'label': None})
            previous_f += recurrent_functions.phi_numpy(f, 0, 2) / 3

        my_plot.set_dict_fig(0,
                             {'xlabel': 'Time $t$', 'ylabel': 'Probability',
                              'title': 'Kernel represented as function of the time'})
        my_plot.set_dict_fig(1,
                             {'xlabel': 'Time event $t_i$', 'ylabel': '',
                              'title': 'Kernel represented as function of events $t_i$'})
        my_plot.show_legend()

    def test_image_CKDE(self):
        nb_of_points = 10000
        xx = np.linspace(-15, 15, nb_of_points)
        mesh = 30. / nb_of_points

        ############################## 3
        ############### left
        zz = np.zeros(nb_of_points)
        max_x = 0.23
        my_plot = plot_functions.APlot(how=(1, 2), sharey=True)
        my_plot.uni_plot(0, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        yy = (recurrent_functions.phi_numpy(xx, -4, 3) + recurrent_functions.phi_numpy(xx, 1, 1)) / 2
        my_plot.uni_plot(0, xx, yy, dict_plot_param={'color': 'r', 'label': 'Kernel for $t_i = 0$'})

        ############### right
        zz = (recurrent_functions.phi_numpy(-xx, -4, 3) + recurrent_functions.phi_numpy(-xx, 1, 1)) / 2
        my_plot.uni_plot(1, [0 for _ in xx],
                         np.linspace(-0.004, max_x, len(xx)),
                         dict_plot_param={'color': 'g', 'label': 'Estimation point',
                                          'linestyle': '--', 'linewidth': 2,
                                          'markersize': 0})
        my_plot.uni_plot(1, xx, zz, dict_plot_param={'color': 'r', 'label': 'Kernel for $t = 0$'})

        my_plot.set_dict_fig(0,
                             {'xlabel': 'Time t', 'ylabel': 'Probability',
                              'title': 'Kernel represented as function of the time'})
        my_plot.set_dict_fig(1,
                             {'xlabel': 'Time event $t_i$', 'ylabel': '',
                              'title': 'Kernel represented as function of events $t_i$'})
        my_plot.show_legend()

    def test_my_geometric_kernel(self):
        # des listes de 50 éléments
        liste_1 = np.full(50, 13)
        liste_2 = np.linspace(2, 24, 50)
        liste_3 = np.append(liste_2[:25], liste_1[25:])
        liste_4 = np.append(np.full(25, 2), np.full(25, 20))
        my_sin = 50 * np.sin(np.linspace(0, 11 / 2 * math.pi, 35)) + 100
        liste_5 = np.append(my_sin, np.full(15, 1))

        my_list = [liste_1, liste_2, liste_3, liste_4, liste_5]
        print(liste_1)
        print(liste_2)
        print(liste_3)
        print(liste_4)
        print(liste_5)

        from scipy.stats.mstats import gmean
        print(my_list)
        for i, l in enumerate(my_list):
            print("{} value is {}. min {} max {}.".format(i, gmean(l), np.quantile(l, 0.15), np.quantile(l, 0.85)))

    def test_form_evol_functions(self):

        list_1 = [constant_parameter, linear_growth, one_jump, moutain_jump, periodic_stop]
        list_2 = [{'constant': 5}, {'a': 2, 'b': 4}, {'when_jump': 0.4, 'original_value': 2, 'new_value': 3},
                  {'when_jump': 0.7, 'a': 2, 'b': 1.5, 'base_value': 0.5},
                  {'base_value': 3, 'a': 1}]

        T_max = 1000
        xx = np.linspace(0, T_max, 100000)
        for fct, param in zip(list_1, list_2):
            fct = np.vectorize(fct)
            yy = fct(xx, T_max=T_max, time_burn_in=0, **param)
            aplot = plot_functions.APlot(how=(1, 1))
            aplot.uni_plot(nb_ax=0, xx=xx, yy=yy, dict_plot_param={"color": "blue", "markersize": 0, "linewidth": 2})
            aplot.set_dict_fig(0, {'title': '', 'xlabel': '', 'ylabel': ''})

        plt.show()
