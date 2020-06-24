##### normal libraries
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import numpy as np #maths library and arrays
import statistics as stat
import pandas as pd #dataframes
import seaborn as sns #envrionement for plots
from matplotlib import pyplot as plt #ploting
import scipy.stats #functions of statistics
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time #allows to time event
import warnings
import math #quick math functions
import cmath  #complex functions

##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions
from classes.class_estimator import *
from classes.class_graph_estimator import *
np.random.seed(124)

##### other files
from class_Estimator_Hawkes import *
from class_kernel import *
import functions_general_for_Hawkes


# defaut kernel, useful for default argument.
kernel_plain = Kernel(fct_kernel=fct_plain, name="flat")
INFINITY = float("inf")


def CDF_exp(x, LAMBDA):
    return - np.log(1 - x) / LAMBDA


def exp_kernel(alpha, beta, t):
    return alpha * np.exp(- beta * t)


def step_fun(tt, time_real):
    # At every index where the jumps occurs and onwards, +1 to the stepfunction.
    y = np.zeros(len(tt))
    for i in range(len(tt)):
        jumps = classical_functions.find_smallest_rank_leq_to_K(time_real, tt[i])
        y[i] = jumps
    return y


class Hawkes_process:
    #### problem with the tt, perhaps think about getting rid of it.
    def __init__(self, tt, parameters):
        print("Creation of a Hawkes Process.")
        print("-" * 78)
        self.tt = tt
        self.parameters = parameters.copy()  # without the copy, if I update the parameters inside HP, it also updates the parameters outside of the object.
        self.ALPHA = self.parameters[1]
        self.BETA = self.parameters[2]
        self.NU = self.parameters[0]
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(self.BETA))
        self.M = np.shape(self.ALPHA)[1]

    def __repr__(self):
        return 'Hawkes process, with parameters : {}, {}, {}'.format(self.NU, self.ALPHA, self.BETA)

    # if plot bool  then draw the path of the simulation.
    def simulation_Hawkes_exact(self, T_max, nb_of_sim=100000,
                                plot_bool=True,
                                silent=True):  # 100 000 is just a safe guard in order to not stuck the computer.
        if not silent: print("Start of the simulation of the Hawkes process.")
        ########################################################################################
        # alpha and beta same shape. Mu a column vector with the initial intensities.
        if nb_of_sim is None and T_max is None:
            print("I need at least one stopping parameter ! I put number of sim to 300.")
            nb_of_sim = 300
        # here alpha and beta should be scalars in a matrix form.
        if np.shape(self.ALPHA) != np.shape(self.BETA):
            raise Exception("Why are the parameters not of the good shape ?")

        ########################################################################################
        # empty vector for stocking the information (the times at which something happens).
        T_t = [[] for _ in range(self.M)]

        # where I evaluate the function of intensity
        intensity = np.zeros((self.M, len(self.tt)))
        last_jump = 0
        # if nb_of_sim is not None :
        counter = 0

        # for the printing :
        last_print = -1

        # For the evaluation, we stock the last lambda. Like aa, (later), each row is a m, each column is a i.

        previous_lambda = np.zeros((self.M, self.M))
        small_lambdas = np.zeros((self.M, self.M, len(self.tt)))

        def CDF_LEE(U, lambda_value, delta):
            if np.asscalar(U) > 1 - np.exp(- lambda_value / delta):
                return INFINITY
            else:
                return -1 / delta * np.log(1 + delta / lambda_value * np.log(1 - U))

        condition = True
        while condition:
            # aa is the matrix of the a_m^i. Each column represents one i, each row a m, just the way the equations are written.
            aa = np.zeros((self.M, self.M + 1))
            ################## first loop over the m_dims.
            ################## second loop over where from.
            for m_dims in range(self.M):
                for i_where_from in range(self.M + 1):
                    U = np.random.rand(1)
                    if i_where_from == 0:
                        aa[m_dims, i_where_from] = CDF_exp(U, self.NU[m_dims])
                    # cases where the other processes can have an impact. If not big enough, it can't: ( spares some computations )
                    elif previous_lambda[i_where_from - 1, m_dims] < 10e-10:
                        aa[m_dims, i_where_from] = INFINITY
                    # cases where it is big enough:
                    else:
                        aa[m_dims, i_where_from] = CDF_LEE(U, previous_lambda[i_where_from - 1, m_dims],
                                                           self.BETA[i_where_from - 1, m_dims])
            # next_a_index indicates the dimension in which the jump happens.
            if self.M > 1:
                # it is tricky : first find where the min is (index) but it is flatten. So I recover coordinates with unravel index.
                # I take [0] bc I only care about where the excitation comes from.
                next_a_index = np.unravel_index(np.argmin(aa, axis=None), aa.shape)[0]
            #otherwise, excitation always from 0 dim.
            else:
                next_a_index = 0
            # min value.
            next_a_value = np.amin(aa)
            previous_jump = last_jump
            last_jump += next_a_value

            # I add the time iff I haven't reached the limit already.
            already_added = False
            if T_max is not None and not already_added:
                already_added = True
                if (last_jump < T_max):
                    T_t[next_a_index].append(last_jump)  # check this is correct
            if nb_of_sim is not None and not already_added:
                if (counter < nb_of_sim - 1):
                    T_t[next_a_index].append(last_jump)  # check this is correct

            # previous lambda gives the lambda for simulation.
            # small lambda is the lambda in every dimension for plotting.
            for ii in range(self.M):
                for jj in range(self.M):
                    if jj == next_a_index:
                        previous_lambda[jj, ii] = previous_lambda[jj, ii] * math.exp(
                            - self.BETA[jj, ii] * next_a_value) + \
                                                            self.ALPHA[jj, ii]
                    else :
                        previous_lambda[jj, ii] = previous_lambda[jj, ii] * math.exp(
                            - self.BETA[jj, ii] * next_a_value)


            if plot_bool:
                # optimize-speed I can search for the index of the last jump. Then, start i_times at this time. It will reduce computational time for high times.
                for i_line in range(self.M):
                    for j_column in range(self.M):
                        for i_times in range(len(self.tt)):
                            # this is when there is the jump. It means the time is exactly smaller but the next one bigger.
                            if self.tt[i_times - 1] <= last_jump and self.tt[i_times] > last_jump:
                                # I filter the lines on which I add the jump. I add the jump to the process iff the value appears on the relevant line of the alpha.
                                if i_line == next_a_index:
                                    small_lambdas[i_line, j_column, i_times] = self.ALPHA[i_line, j_column] * np.exp(
                                        - self.BETA[i_line, j_column] * (self.tt[i_times] - last_jump))
                                # since we are at the jump, one doesn't have to look further.
                                break
                                # the window of times I haven't updated.
                            # I am updating all the other times.
                            if self.tt[i_times] > previous_jump and self.tt[i_times] < last_jump:
                                small_lambdas[i_line, j_column, i_times] += small_lambdas[
                                                                                i_line, j_column, i_times - 1] * np.exp(
                                    - self.BETA[i_line, j_column] * (self.tt[i_times] - self.tt[i_times - 1]))

            # condition part:
            if nb_of_sim is not None:
                counter += 1
                # print part
                if counter == 1:
                    if not silent:
                        print("Beginning of the simulation.")
                if counter % 5000 == 0:
                    if not silent:
                        print("Jump {} out of total number of jumps {}.".format(counter, nb_of_sim))
                # condition :
                if not (counter < nb_of_sim - 1):
                    condition = False

            if T_max is not None:
                # print part
                if round(last_jump, -1) % 500 == 0 and round(last_jump, -1) != last_print:
                    last_print = round(last_jump, -1)
                    if not silent:
                        print("Time {} out of total time : {}.".format(round(last_jump, -1), T_max))
                if not (last_jump < T_max):
                    condition = False
        # will be an empty list if not for plot purpose.
        if plot_bool:
            for i_line in range(self.M):
                for i_times in range(len(self.tt)):
                    intensity[i_line, i_times] = self.NU[i_line]
                    for j_from in range(self.M):
                        intensity[i_line, i_times] += small_lambdas[j_from, i_line, i_times]
        # tricks, not giving back a list of list but a list of numpy array.
        # T_t = [np.array(aa) for aa in T_t]
        return intensity, T_t

    # BIANCA-HERE this guy shouldn't be here... one has to move it to graphs.
    def plot_hawkes(self, time_real, intensity, name=None):
        # I need alpha and beta in order for me to plot them.
        shape_intensity = np.shape(intensity)
        plt.figure(figsize=(10, 5))
        x = self.tt
        # colors :
        color = iter(plt.cm.rainbow(np.linspace(0, 1, shape_intensity[0])))
        upper_ax = plt.subplot2grid((21, 21), (0, 0), rowspan=14,
                                    colspan=16)  # plt.subplot2grid((21, 21), (0, 0), rowspan=15, colspan=10)
        lower_ax = plt.subplot2grid((21, 21), (16, 0), rowspan=8, colspan=16)
        for i_dim in range(shape_intensity[0]):
            # the main
            c = next(color)
            y = intensity[i_dim, :]
            number_on_display = i_dim + 1
            label_plot = str(" dimension " + str(number_on_display))
            upper_ax.plot(x, y, 'o-', markersize=0.2, linewidth=0.4, label=label_plot, color=c)
            upper_ax.set_ylabel("Intensity : $\Lambda (t)$")
            # the underlying
            y = 4 * i_dim + step_fun(x, np.array(time_real[i_dim]))
            lower_ax.plot(x, y, 'o-', markersize=0.5, linewidth=0.5, color=c)
            lower_ax.set_xlabel("Time")
            lower_ax.set_ylabel("Point Process : $N_t$")

        upper_ax.legend(loc='best')
        upper_ax.grid(True)
        lower_ax.grid(True)
        # Matrix plot :
        plt.subplot2grid((21, 21), (1, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\alpha$", fontsize=12, color='black')
        plt.axis('off')
        ax = plt.subplot2grid((21, 21), (3, 16), rowspan=5, colspan=5)
        im = plt.imshow(self.ALPHA, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(self.ALPHA):
            ax.text(i, j, label, ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (9, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\beta$", fontsize=12, color='black')
        plt.axis('off')
        ax = plt.subplot2grid((21, 21), (10, 16), rowspan=5, colspan=5)
        im = plt.imshow(self.BETA, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(self.BETA):
            ax.text(i, j, label, ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (19, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\mu$" + str(self.NU), fontsize=11, color='black')
        plt.axis('off')

        if name is not None:
            string = "Hawkes_simulation_" + name + ".png"
            plt.savefig(string, dpi=1000)
        else:
            plt.savefig("Hawkes_simulation.png", dpi=1000)

        return

    def update_coef(self, time, fct, **kwargs):
        # fct here is a list of lists of lists; because we want to change each coeff indep.
        # for NU, the functions are on the first column.
        for i in range(self.M):
            self.NU[i] = (fct[0][i])(time, **kwargs)
            for j in range(self.M):
                self.ALPHA[i, j] = (fct[1][i][j])(time, **kwargs)
                self.BETA[i, j] = (fct[2][i][j])(time, **kwargs)

        return
