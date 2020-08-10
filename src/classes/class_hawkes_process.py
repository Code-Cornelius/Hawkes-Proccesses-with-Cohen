##### normal libraries

##### my libraries
from classes.class_graph_estimator import *

##### other files
from classes.class_kernel import *
from functions_general_for_Hawkes import multi_list_generator

np.random.seed(124)


# section ######################################################################
#  #############################################################################
 # fcts


# defaut kernel, useful for default argument.
kernel_plain = Kernel(fct_kernel=fct_plain, name="flat")
INFINITY = float("inf")


def CDF_exp(x, LAMBDA):
    return - np.log(1 - x) / LAMBDA # I inverse 1-x bc the uniform can be equal to 0, not defined in the log.


def CDF_LEE(U, lambda_value, delta):
    if U.item() > 1 - np.exp(- lambda_value / delta):
        return INFINITY
    else:
        return -1 / delta * np.log(1 + delta / lambda_value * np.log(1 - U))


def exp_kernel(alpha, beta, t):
    return alpha * np.exp(- beta * t)


def lewis_non_homo(max_time, actual_time, max_nu, fct_value, **kwargs):
    '''

    Args:
        max_time: in order to avoid infinite loop.
        actual_time: current time, in order to update the parameter NU
        max_nu:  over the interval for thinning.
        fct_value: NU fct.
        **kwargs: for NU fct.

    Returns:

    '''
    arrival_time = 0
    while actual_time + arrival_time < max_time:
        U = np.random.rand(1)
        arrival_time += CDF_exp(U, max_nu)
        D = np.random.rand(1)
        if (D <= fct_value(actual_time + arrival_time, **kwargs) / max_nu):
            return arrival_time
    return INFINITY


def step_fun(tt, time_real):
    # At every index where the jumps occurs and onwards, +1 to the step-function.
    y = np.zeros(len(tt))
    for i in range(len(tt)):
        jumps = classical_functions.find_smallest_rank_leq_to_K(time_real, tt[i])
        y[i] = jumps
    return y

# section ######################################################################
#  #############################################################################
 # class

class Hawkes_process:
    # how to acces class fields?
    time_burn_in = 100
    nb_points_burned = 6000
    points_burned = np.linspace(0, time_burn_in, nb_points_burned)
    #### problem with the tt, perhaps think about getting rid of it.
    def __init__(self, the_update_functions):
        print("Creation of a Hawkes Process.")
        print("-" * 78)

        self.the_update_functions = the_update_functions.copy()  # without the copy, if I update the the_update_functions inside HP, it also updates the the_update_functions outside of the object.
        self.ALPHA = self.the_update_functions[1]
        self.BETA = self.the_update_functions[2]
        self.NU = self.the_update_functions[0]
        self.parameters_line = np.append(np.append(self.NU, np.ravel(self.ALPHA)), np.ravel(self.BETA))
        self.M = np.shape(self.ALPHA)[1]
        self.plot_parameters_hawkes()

    def __call__(self, t, T_max):
        NU, ALPHA, BETA = multi_list_generator(self.M)
        for i in range( self.M ):
            NU[i] = self.NU[i](t, T_max, Hawkes_process.time_burn_in)
            for j in range( self.M ):
                ALPHA[i][j] =self.ALPHA[i][j](t, T_max, Hawkes_process.time_burn_in)
                BETA[i][j] =  self.BETA[i][j](t, T_max, Hawkes_process.time_burn_in)

        return f'a Hawkes process, with parameters at time {t} : {NU}, {ALPHA}, {BETA}'

    def __repr__(self):
        return self.__call__(0, 1000)


    def plot_parameters_hawkes(self):
        # I m printing the evolution of the parameters there.
        aplot = APlot(how=(1, self.M))
        tt = np.linspace(0,1,1000)
        my_colors = plt.cm.rainbow(np.linspace(0, 1, 2*self.M))
        for i_dim in range(self.M):
            xx_nu = [self.NU[i_dim](t, 1, 0) for t in tt]
            aplot.uni_plot(nb_ax=i_dim, yy=xx_nu, xx=tt, dict_plot_param={"label": f"nu, {i_dim}", "color": "blue", "markersize": 0, "linewidth" : 2}, tight = False)
            color = iter(my_colors)
            for j_dim in range(self.M):
                c1 = next(color)
                c2 = next(color)
                xx_alpha = [self.ALPHA[i_dim][j_dim](t, 1, 0) for t in tt]
                xx_beta = [self.BETA[i_dim][j_dim](t, 1, 0) for t in tt]
                aplot.uni_plot(nb_ax=i_dim, yy=xx_alpha, xx=tt, dict_plot_param = {"label": f"alpha, {i_dim},{j_dim}.", "color": c1, "markersize": 0, "linewidth" : 2}, tight = False)
                aplot.uni_plot(nb_ax=i_dim, yy=xx_beta, xx=tt, dict_plot_param = {"label": f"beta, {i_dim},{j_dim}.", "color": c2, "markersize": 0, "linewidth" : 2}, tight = False)

                aplot.set_dict_fig(i_dim, {'title': "Evolution of the parameters, time in $\%$ of total; dimension : {}".format(i_dim), 'xlabel':'', 'ylabel':''})
            aplot.show_legend(i_dim)


    # if plot bool  then draw the path of the simulation.
    def simulation_Hawkes_exact_with_burn_in(self, tt, nb_of_sim=100000,
                                plot_bool=True,
                                silent=True):  # 100 000 is just a safe guard in order to not stuck the computer.

        #I want to add burn in, je dois simuler plus longtemps et effacer le début.
        #en gros je fais une simul sur T + un param, genre 100, et je cherche intensity et jump après 100 jusqu'à T+100.
        tt_burn = np.append(Hawkes_process.points_burned, tt + Hawkes_process.time_burn_in)
        T_max = tt[-1]


        if not silent: print("Start of the simulation of the Hawkes process.")
        ########################################################################################
        # alpha and beta same shape. Mu a column vector with the initial intensities.

        # old conditions
        # if nb_of_sim is None and T_max is None:
        #     print("I need at least one stopping parameter ! I put number of sim to 300.")
        #     nb_of_sim = 300


        # here alpha and beta should be scalars in a matrix form.
        if np.shape(self.ALPHA) != np.shape(self.BETA):
            raise Exception("Why are the the_update_functions not of the good shape ?")

        ########################################################################################
        # empty vector for stocking the information (the times at which something happens).
        T_t = [[] for _ in range(self.M)]

        # where I evaluate the function of intensity
        intensity = np.zeros((self.M, len(tt_burn)))
        last_jump = 0
        # if nb_of_sim is not None :
        counter = 0

        # for the printing :
        last_print = -1

        # For the evaluation, we stock the last lambda. Like aa, (later), each row is a m, each column is a i.
        # previous_lambda is the old intensity, and we have the small_lambdas, the each component of the intensity.
        # We don't need to incorporate the burned point in it. It appears in the previous lambda.
        previous_lambda = np.zeros((self.M, self.M))
        small_lambdas = np.zeros((self.M, self.M, len(tt_burn)))

        condition = True

        #I need the max value of mu for thinning simulation:
        # it is an array with the max in each dimension.
        the_funct_nu = [np.vectorize(self.NU[i]) for i in range(self.M)]
        max_nu = [np.max(the_funct_nu[i](tt_burn, T_max, Hawkes_process.time_burn_in)) for i in range(self.M)]
        while condition:
            # aa is the matrix of the a_m^i. Each column represents one i, each row a m, just the way the equations are written.
            aa = np.zeros((self.M, self.M + 1))
            ################## first loop over the m_dims.
            ################## second loop over where from.
            for m_dims in range(self.M):
                for i_where_from in range(self.M + 1):
                    if i_where_from == 0:
                        # todo change function MU
                        aa[m_dims, i_where_from] = lewis_non_homo( T_max + Hawkes_process.time_burn_in,
                                                                  last_jump,
                                                                  max_nu[m_dims],
                                                                  self.NU[m_dims],
                                                                  T_max = T_max,
                                                                  time_burn_in = Hawkes_process.time_burn_in )
                    # cases where the other processes can have an impact. If not big enough, it can't: ( spares some computations )
                    elif previous_lambda[i_where_from - 1, m_dims] < 10e-10:
                        aa[m_dims, i_where_from] = INFINITY
                    # cases where it is big enough:
                    else:
                        U = np.random.rand(1)
                        # todo change function BETA
                        aa[m_dims, i_where_from] = CDF_LEE(U, previous_lambda[i_where_from - 1, m_dims],
                                                           self.BETA[i_where_from - 1][ m_dims](0,T_max, Hawkes_process.time_burn_in))
            # next_a_index indicates the dimension in which the jump happens.
            if self.M > 1:
                # it is tricky : first find where the min is (index) but it is flatten. So I recover coordinates with unravel index.
                # I take [0] bc I only care about where the excitation comes from.
                next_a_index = np.unravel_index(np.argmin(aa, axis=None), aa.shape)[0]
            # otherwise, excitation always from 0 dim.
            else:
                next_a_index = 0
            # min value.
            next_a_value = np.amin(aa)


            # last jump is the time at which the current interesting jump happened.
            # previous_jump is the one before.
            previous_jump = last_jump
            last_jump += next_a_value

            if not silent: print("actual jump : ", last_jump)

            # I add the time iff I haven't reached the limit already.

            # the already added is useful only for the double possibility between T_max and nb_of_sim.
            #       already_added = False

            if T_max is not None : #and not already_added:
                # already_added = True
                if (last_jump < T_max + Hawkes_process.time_burn_in):
                    T_t[next_a_index].append(last_jump)
            # if nb_of_sim is not None and not already_added:
            #     if (counter < nb_of_sim - 1):
            #         T_t[next_a_index].append(last_jump)  # check this is correct



            # previous lambda gives the lambda for simulation.
            # small lambda is the lambda in every dimension for plotting.
            for ii in range(self.M):
                for jj in range(self.M):
                    if jj == next_a_index:
                        # todo change function ALPHA BETA
                        previous_lambda[jj, ii] = previous_lambda[jj, ii] * math.exp(
                            - self.BETA[jj][ii](last_jump, T_max, Hawkes_process.time_burn_in) * next_a_value) + \
                                                  self.ALPHA[jj][ii](last_jump, T_max, Hawkes_process.time_burn_in)
                    else:
                        # todo change function BETA
                        previous_lambda[jj, ii] = previous_lambda[jj, ii] * math.exp(
                            - self.BETA[jj][ii](last_jump, T_max, Hawkes_process.time_burn_in) * next_a_value)


            if plot_bool:
                # print("previous : ", previous_jump)
                # print("last : ", last_jump)
                first_index_time = classical_functions.find_smallest_rank_leq_to_K(tt_burn, previous_jump)
                for i_line in range(self.M):
                    for j_column in range(self.M):
                        for i_times in range(first_index_time, len(tt_burn)):
                            # this is when there is the jump. It means the time is exactly smaller but the next one bigger.
                            if tt_burn[i_times - 1] <= last_jump and tt_burn[i_times] > last_jump:
                                # I filter the lines on which I add the jump. I add the jump to the process iff the value appears on the relevant line of the alpha.
                                if i_line == next_a_index:
                                    # todo change function ALPHA BETA
                                    small_lambdas[i_line, j_column, i_times] = self.ALPHA[
                                                                                   i_line][j_column](last_jump, T_max, Hawkes_process.time_burn_in) * np.exp(
                                        - self.BETA[i_line][j_column](last_jump, T_max, Hawkes_process.time_burn_in) * (tt_burn[i_times] - last_jump))
                                # since we are at the jump, one doesn't have to look further.
                                # break is going out of time loop.
                                break
                            # the window of times I haven't updated.
                            # I am updating all the other times.
                            # todo change function BETA
                            if tt_burn[i_times]  > previous_jump and tt_burn[i_times] < last_jump:
                                small_lambdas[i_line, j_column, i_times] += small_lambdas[
                                                                                i_line, j_column, i_times - 1] * np.exp(
                                    - self.BETA[i_line][j_column](last_jump, T_max, Hawkes_process.time_burn_in) * (tt_burn[i_times] - tt_burn[i_times - 1]))

            # condition part:
            if nb_of_sim is not None:
                counter += 1
                # print part
                if counter % 5000 == 0:
                    if not silent:
                        print(f"Jump {counter} out of total number of jumps {nb_of_sim}.")
                # condition :
                if not (counter < nb_of_sim - 1):
                    condition = False

            if T_max is not None:
                # print part
                if not silent:
                    if round(last_jump, -1) % 800 == 0 and round(last_jump, -1) != last_print:
                        last_print = round(last_jump, -1)
                        print(f"Time {round(last_jump, -1)} out of total time : {T_max}.")
                # IF YOU ARE TOO BIG IN TIME:
                # I add the burn in
                if not (last_jump < T_max + Hawkes_process.time_burn_in):
                    condition = False
        # will be an empty list if not for plot purpose.
        if plot_bool:
            for i_line in range(self.M):
                for counter_times, i_times in enumerate(tt_burn):
                    # todo change function NU
                    intensity[i_line, counter_times] = self.NU[i_line](i_times, T_max, Hawkes_process.time_burn_in)
                    for j_from in range(self.M):
                        intensity[i_line, counter_times] += small_lambdas[j_from, i_line, counter_times]


        if not silent: print("inside not shifted : ", T_t)
        #conditions on the times, we want a subset of them.


        # intensity bis is the truncated version of intensity.
        intensity_bis = np.zeros((self.M, len(tt_burn) - Hawkes_process.nb_points_burned))
        for i in range(len(T_t)):
            # find the times big enough.
            i_time = classical_functions.find_smallest_rank_leq_to_K(np.array(T_t[i]), Hawkes_process.time_burn_in)
            # shift the times
            T_t[i]= list(
                np.array(   T_t[i][i_time:] ) - Hawkes_process.time_burn_in
                        )
            intensity_bis[i,:] = list(
                np.array(   intensity[i][Hawkes_process.nb_points_burned:] )
                        )


        # tricks, not giving back a list of list but a list of numpy array.
        # T_t = [np.array(aa) for aa in T_t]
        return intensity_bis, T_t







    def plot_hawkes(self, tt, time_real, intensity, name=None):

        NU, ALPHA, BETA = multi_list_generator(self.M)
        for i in range( self.M ):
            NU[i] = self.NU[i](0, 1000,0)
            for j in range( self.M ):
                ALPHA[i][j] = self.ALPHA[i][j](0, 1000, 0)
                BETA[i][j]  = self.BETA[i][j](0, 1000, 0)


        # I need alpha and beta in order for me to plot them.
        shape_intensity = np.shape(intensity)
        plt.figure(figsize=(10, 5))
        x = tt
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
            upper_ax.set_ylabel("Intensity : $\lambda (t)$")
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
        im = plt.imshow(ALPHA, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(ALPHA):
            ax.text(i, j, label, ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (9, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\beta$", fontsize=12, color='black')
        plt.axis('off')
        ax = plt.subplot2grid((21, 21), (10, 16), rowspan=5, colspan=5)
        im = plt.imshow(BETA, cmap="coolwarm")
        for (j, i), label in np.ndenumerate(BETA):
            ax.text(i, j, label, ha='center', va='center')
        plt.colorbar(im)
        plt.axis('off')

        plt.subplot2grid((21, 21), (19, 16), rowspan=1, colspan=5)
        plt.text(0.5, 0, "$\\nu = $ " + str(NU), fontsize=11, color='black')
        plt.axis('off')

        if name is not None:
            string = "Hawkes_simulation_" + name + ".png"
            plt.savefig(string, dpi=1000)
        else:
            plt.savefig("Hawkes_simulation.png", dpi=1000)

        return

# section ######################################################################
#  #############################################################################
# test


