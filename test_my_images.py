# normal libraries
import unittest

import numpy as np  #maths library and arrays
import statistics as stat
import pandas as pd  #dataframes
import seaborn as sns  #envrionement for plots
from matplotlib import pyplot as plt  #ploting 
import scipy.stats  #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time  #allows to time event
import warnings
import math  #quick math functions
import cmath  #complex functions

# my libraries
import classical_functions
import decorators_functions
import financial_functions
import functions_networkx
import plot_functions
import recurrent_functions
import errors.error_convergence
import classes.class_estimator
import classes.class_graph_estimator

np.random.seed(124)
# other files

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("hello biatch")

print("bouhouhou")
class Test_images(unittest.TestCase):

    def tearDown(self):
        plt.show()

    def test_image_different_kernel_vision(self):
        xx = np.linspace( -10,10, 1000)
        for f in [-7,-6,3,4,5]:
            print(f)
            yy = recurrent_functions.phi_numpy(xx, f,2)
            print(yy)
            plot_functions.APlot(datax = xx, datay = yy)
            plt.show()


    def test_test(self):
        print(42)
