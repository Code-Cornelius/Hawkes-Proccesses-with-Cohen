from unittest import TestCase


class Test(TestCase):
    def test_my_rescale_sin(self):
        self.fail()

    def test_akde_scaling(self):
        self.fail()

    def test_rescale_min_max(self):
        self.fail()

    def test_check_evoluating(self):
        self.fail()

    def test_rescaling_kernel_processing(self):
        self.fail()

    def test_creator_list_kernels(self):
        self.fail()

    def test_creator_kernels_adaptive(self):
        self.fail()



# section ######################################################################
#  #############################################################################
# test


########### test adaptive window
T_t = [np.linspace(0.1, 100, 10000)]
G = 10.
# T_t = [np.random.randint(0,6*G, 20)]
eval_point = [0]
for i in eval_point:
    res = my_rescale_sin(value_at_each_time = T_t, L=0.02, R=0.75)
    aplot = APlot(how=(1, 1))
    aplot.uni_plot(nb_ax=0, xx=T_t[0], yy=res[0])
    aplot.plot_vertical_line(G, np.linspace(-5, 105, 1000), nb_ax=0,
                             dict_plot_param={'color': 'k', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                              'label': 'geom. mean'})
    aplot.plot_vertical_line(min, np.linspace(-5, 105, 1000), nb_ax=0,
                             dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                              'label': 'lower bound'})
    aplot.plot_vertical_line(max, np.linspace(-5, 105, 1000), nb_ax=0,
                             dict_plot_param={'color': 'g', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                              'label': 'upper bound'})
    aplot.set_dict_fig(0,
                       {'title': 'Adaptive scaling for Adaptive Window Width', 'xlabel': 'Value', 'ylabel': 'Scaling'})
    aplot.show_legend()

eval_point = [0]
for _ in eval_point:
    res = AKDE_scaling(T_t, G, gamma=0.5)
    aplot = APlot(how=(1, 1))
    aplot.uni_plot(nb_ax=0, xx=T_t[0], yy=res[0])
    aplot.plot_vertical_line(G, np.linspace(-1, 10, 1000), nb_ax=0,
                             dict_plot_param={'color': 'k', 'linestyle': '--', 'markersize': 0, 'linewidth': 2,
                                              'label': 'geom. mean'})
    aplot.set_dict_fig(0,
                       {'title': 'Adaptive scaling for Adaptive Window Width', 'xlabel': 'Value', 'ylabel': 'Scaling'})
    aplot.show_legend()