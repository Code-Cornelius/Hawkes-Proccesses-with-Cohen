##### normal libraries
import numpy as np
import statistics as stat
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats
from inspect import signature
from operator import itemgetter  # at some point I need to get the list of ranks of a list.
import time
import bisect

##### my libraries
import plot_functions
import decorators_functions
import classical_functions
import recurrent_functions
from decorators_functions import Memoization
from classes.class_estimator import *
from classes.class_graph_estimator import *

##### other files
from class_Estimator_Hawkes import *
from class_Graph_Estimator_Hawkes import *
from class_hawkes_process import *
from class_kernel import *
import functions_change_point_analysis
import functions_MLE
import functions_fct_evol_parameters
import functions_general_for_Hawkes



@Memoization(key_names=['m', 'k'])
def denomR(m, k, T_t, ALPHA, BETA, MU):
    constant = 0
    _, M = np.shape(ALPHA)
    for j in range(M): # no need to optimise because M is small
        constant += ALPHA[m, j] * R(m=m, n=j, k=k + 1, T_t=T_t, BETA=BETA)  # +1 for i, starts at 1.
    return (MU[m] + constant)


# R, dont forget to put k as k+1 in the loops
# recursive, memoization
@Memoization(key_names=['m', 'n', 'k'])
def R(m, n, k, T_t, BETA, end=-10):
    constant = 0
    # M AND N STARTS AT 0 AND FINISH AT M-1.
    if k < 1:
        raise Exception("YOu moRoN")
    if k == 1:
        if m == n:
            return 0
        else:  # end exists and has been given.
            i = 0
            while i < len(T_t[n]) and T_t[n][i] < T_t[m][0]:
                constant += np.exp(-BETA[m, n] * (T_t[m][0] - T_t[n][i]))
                i += 1
            return constant
    # here I compute the value.
    first_coeff = np.exp(-BETA[m, n] * (T_t[m][k - 1] - T_t[m][k - 2]))

    if m == n:
        return first_coeff * (1 + R(m=m, n=n, k=k - 1, T_t=T_t, BETA=BETA))  # since m = n you don't care about end.

    # if = -10, it means it s the first time I compute R for that m and n.
    if end == -10:
        # bisect left is perfect for that. If I have
        # [3,4,5,6,7] ask for 3 I get 0, ask for 3.1 I get 1.
        # PROBLEM !!! in case: [3,5] ; [2,4] ask for 3 I get -1.
        #
        end = bisect.bisect_left(T_t[n], T_t[m][k - 1]) - 1

    # if end == - 1 it means there is no value inside the list n that is bigger than t_i^m.
    if end == -1:
        return 0

    # thats the sum on the right. We are above the inequalities so we only check for the lower bound.
    for i in range(end, -1, -1):
        if T_t[m][k - 2] <= T_t[n][i]:
            constant += np.exp(-BETA[m, n] * (T_t[m][k - 1] - T_t[n][i]))
        else:
            # end of vector
            break
    return first_coeff * R(m=m, n=n, k=k - 1, T_t=T_t, BETA=BETA, end=i) + constant


# R, dont forget to put k as k+1 in the loops
# closed form of R
@Memoization(key_names=['m', 'n', 'k'])
def compute_R2(m, n, k, T_t, BETA, end=10):
    # M AND N STARTS AT 0 AND FINISH AT M-1.
    if k < 1:
        raise Exception("YOu moRoN")
    if k == 1:
        return 0
    constant = 0
    for i in range(len(T_t[n])):
        if T_t[n][i] < T_t[m][k - 1]:
            constant += np.exp(-BETA[m, n] * (T_t[m][k - 1] - T_t[n][i]))
        else:
            break
    return constant


def compute_R_dashes(m, n, T_t, BETA, end=0):
    matrix_diff = np.maximum(np.subtract.outer(np.array(T_t[m])
                                               , np.array(T_t[n])),
                             0)
    dashes = matrix_diff * np.exp(-BETA[m, n] * matrix_diff)
    dash_dashes = dashes * matrix_diff
    # np.power is not efficient for scalars, but good looking.
    previous_Rs_dash[(m, n)] = list(dashes.sum(axis=1))

    previous_Rs_dash_dash[(m, n)] = list(dash_dashes.sum(axis=1))


previous_Rs_dash = {}
# here the k has to be shifted
def get_R_dash(m, n, k, T_t, BETA, end=-10):
    # go from 1 to the number jumps included. However, compute gives back shifted. From 0 to number jumps excluded. So the bounding has to be done here.
    if (m, n) not in previous_Rs_dash:
        compute_R_dashes(m, n, T_t, BETA, end=end)

    return previous_Rs_dash[(m, n)][k-1]


previous_Rs_dash_dash = {}
# here the k has to be shifted
def get_R_dash_dash(m, n, k, T_t, BETA, end=-10):
    # go from 1 to the number jumps included. However, compute gives back shifted. From 0 to number jumps excluded. So the bounding has to be done here.
    if (m, n) not in previous_Rs_dash_dash:
        compute_R_dashes(m, n, T_t, BETA, end=end)
    return previous_Rs_dash_dash[(m, n)][k - 1]



# first derivative
def del_L_mu(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = - T + np.sum(w[m] * np.reciprocal(vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_alpha(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    # 1
    my_jumps = np.array(T_t[n])
    ans1 = np.sum(w[n] * (1 - np.exp(- BETA[m, n] * (T - my_jumps))))
    # 2
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    vector_R = np.array([R(m=m, n=n, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    ans2 = np.sum(w[m] * vector_R * np.reciprocal(vector_denomR))  # in denomR the i is already shifted.
    return -1 / BETA[m, n] * ans1 + ans2


def del_L_beta(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    # my_time = time.time()
    _, M = np.shape(ALPHA)

    my_jumps = T - np.array(T_t[n])
    ANS1 = np.sum(w[n] * (1 - np.exp(- BETA[m, n] * (my_jumps))))
    ANS2 = np.sum(w[n] * (my_jumps * np.exp(- BETA[m, n] * (my_jumps))))

    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    ANS3 = ALPHA[m, n] * np.sum(
        w[m] * vector_R_dash * np.reciprocal(vector_denomR))  # in denomR the i is already shifted.

    # time_computational(my_time, time.time(), title="beta del")
    return ALPHA[m, n] / (BETA[m, n] * BETA[m, n]) * ANS1 - ALPHA[m, n] / (BETA[m, n]) * ANS2 - ANS3






























# second derivatives
def del_L_mu_mu(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    # my_time = time.time()
    _, M = np.shape(ALPHA)

    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = -  np.sum(w[m] * np.reciprocal(vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_mu_mu_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0


def del_L_alpha_alpha(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_R = np.array([R(m=m, n=n, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = -  np.sum(w[m] * vector_R * vector_R * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_alpha_alpha_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_R = np.array([R(m=m, n=n, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([R(m=m, n=n_dash, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = -  np.sum(w[m] * vector_R * vector_R_dash * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_alpha_alpha_dif_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0


def del_L_alpha_mu(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_R = np.array([R(m=m, n=n, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = -  np.sum(
        w[m] * vector_R * np.reciprocal(vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_alpha_mu_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0


def del_L_beta_mu(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ans = ALPHA[m, n] * np.sum(
        w[m] * vector_R_dash * np.reciprocal(vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ans


def del_L_beta_mu_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0


def del_L_beta_alpha(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)

    my_jumps = T - np.array(T_t[n])
    ANS1 = np.sum(w[n] * (my_jumps * np.exp(- BETA[m, n] * (my_jumps))))
    ANS1 *= -1 / BETA[m, n]
    ANS2 = np.sum(w[n] *
                  (1 - np.exp(- BETA[m, n] * (my_jumps)))
                  )
    ANS2 *= 1 / BETA[m, n] / BETA[m, n]

    vector_R = np.array([R(m=m, n=n, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ANS3 = - np.sum(w[m] * vector_R_dash * np.reciprocal(vector_denomR))  # in denomR the i is already shifted.
    ANS4 = ALPHA[m, n] * np.sum(w[m] * vector_R_dash * vector_R * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ANS1 + ANS2 + ANS3 + ANS4


def del_L_beta_alpha_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)

    vector_R = np.array([R(m=m, n=n_dash, k=i + 1, T_t=T_t, BETA=BETA) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array(
        [denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])  # in denomR the i is already shifted.
    ANS = ALPHA[m, n] * np.sum(w[m] * vector_R_dash * vector_R * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ANS


def del_L_beta_alpha_dif_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0


def del_L_beta_beta(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    B = BETA[m, n]
    my_jumps = T - np.array(T_t[n])
    ANS1 = np.sum(w[n] * (1 - np.exp(- B * (my_jumps))))
    ANS1 *= - 2 * ALPHA[m, n] / (B * B * B)
    ANS2 = np.sum(w[n] * (my_jumps * np.exp(- B * (my_jumps))))
    ANS2 *= 2 * ALPHA[m, n] / (B * B)
    ANS3 = np.sum(w[n] * (my_jumps * my_jumps * np.exp(- B * (my_jumps))))
    ANS3 *= ALPHA[m, n] / (B)

    vector_R_dash_dash = np.array([get_R_dash_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])

    ANS4 = ALPHA[m, n] * np.sum(
        w[m] * vector_R_dash_dash * np.reciprocal(vector_denomR))  # in denomR the i is already shifted.
    ANS4 -= ALPHA[m, n] * ALPHA[m, n] * np.sum(w[m] * vector_R_dash * vector_R_dash * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ANS1 + ANS2 + ANS3 + ANS4


def del_L_beta_beta_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    vector_R_dash_dash = np.array([get_R_dash(m, n_dash, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_R_dash = np.array([get_R_dash(m, n, i + 1, T_t, BETA) for i in range(len(T_t[m]))])
    vector_denomR = np.array([denomR(m=m, k=i, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU) for i in range(len(T_t[m]))])
    ANS = - ALPHA[m, n] * ALPHA[m, n_dash] * np.sum(w[m] * vector_R_dash * vector_R_dash_dash * np.reciprocal(
        vector_denomR * vector_denomR))  # in denomR the i is already shifted.
    return ANS


def del_L_beta_beta_dif_dif(m, n, n_dash, T_t, ALPHA, BETA, MU, T, w):
    return 0









































# global functions for MLE
def special_matrix_creator_rect(size, function_diag, function_sides, **kwargs):
    matrix = np.zeros((size, size * size))
    for ii in range(size):
        for jj in range(size * size):
            if ii * size - 1 < jj and jj < (ii + 1) * size:
                matrix[ii, jj] = function_diag(m=jj // size,
                                               n=jj % size,
                                               n_dash=0, # no n_dash...
                                               **kwargs)
            else:
                matrix[ii, jj] = function_sides(m=jj // size,
                                                n=jj % size,
                                                n_dash=0, # no n_dash...
                                                **kwargs)
    return matrix


# This is for the diagonal of the second order derivative
def special_matrix_creator_square(size, function_diag, function_sides, function_wings, **kwargs):
    # RED is creating the elements of the diagonal of the matrix of matrices.
    def red(size, function_diag, function_sides, i, j, **kwargs):
        # i and j are the top left element nb
        matrix = np.zeros((size, size))
        for ii in range(size):
            for jj in range(size):
                if ii == jj:
                    matrix[ii, jj] = function_diag(n=i + ii, n_dash=j, **kwargs) # no need for n_dash in that case...
                #work-in-progress
                #elif ii < jj:
                # work-in-progress
                else :
                    matrix[ii, jj] = function_sides(n=i + ii, n_dash=j + jj, **kwargs)
        return matrix

    # I GO THROUGHT THE MATRIX BLUE. I discriminate diagonal and upper triangular matrix.
    # THe lower triangular matrix is filled by transposition

    ## ii and jj go through the big blocks of the matrix with reds on the diagonal and black else where.
    matrix = np.zeros((size ** 2, size ** 2))
    for ii in range(size):
        for jj in range(size):
            if size * jj + size * ii < size * size:
                if jj == 0:
                    matrix[size * ii:size * ii + size, size * ii + size * jj:size * ii + size * jj + size] = red(
                        size, function_diag, function_sides, (size * ii) % size, (size * jj + size * ii) % size, m=ii,
                        **kwargs)
                elif ii < jj:
                    # since we re not in the diagonals blocks, we iterate through the elements of the blocks
                    for iii in range(size):
                        for jjj in range(size):  # size * ii + iii , size * ii + size * jj + jjj,
                            matrix[size * ii + iii, size * ii + size * jj + jjj] = function_wings(jj, jjj,
                                                                                                  n_dash=ii * size + iii,
                                                                                                  **kwargs)
    # work-in-progress:
    # I DONT WANT TO COPY BC APART FROM DIAG, ELEMENTS ARE NOT EQUAL
    # copy upper and lower part
    # for ii in range(size * size):
    #     for jj in range(size * size):
    #         if jj < ii:
    #             matrix[ii, jj] = matrix[jj, ii]
    # work-in-progress:

    return matrix



















# function usef inside first_derivative for clearing all memoization's dict.
def dict_clear():
    R.clear()
    previous_Rs_dash.clear()
    previous_Rs_dash_dash.clear()
    denomR.clear()
    return


# the 3 functions
def likelihood(T_t, ALPHA, BETA, MU, T):
    dict_clear()
    ans = 0
    M = len(T_t)
    # len(T_t) should be the number of dimensions, and the fol. sum correspond to every single likelihood
    # in every single dimension, L^M
    for i in range(M):
        # re initialization of value_i, putting it to initial value
        value_i = - MU[i] * T
        for j in range(M):
            sum_value = 0
            # we go through all values inside i-th list of jumps
            for k in range(len(T_t[j])):
                sum_value = 1 - np.exp(- BETA[i, j] * (T - T_t[j][k]))
            value_i -= ALPHA[i, j] / BETA[i, j] * sum_value
        for k in range(len(T_t[i])):
            inside_big_third_sum = MU[i]
            for j in range(M):
                inside_big_third_sum += ALPHA[i, j] * R(m=i, n=j, k=k + 1, T_t=T_t, BETA=BETA)
        value_i += np.log(inside_big_third_sum)
        ans += value_i
    return ans


# return del Mu, del ALPHA, del BETA
def first_derivative(T_t, ALPHA, BETA, MU, T, w):
    dict_clear()

    _, M = np.shape(ALPHA)
    A = np.array([])
    B = np.array([])
    C = np.array([])
    for i in range(M):
        A = np.append(A, del_L_mu(i, 0, 0, T_t, ALPHA, BETA, MU, T, w))
    for i in range(M):
        for j in range(M):
            B = np.append(B, del_L_alpha(i, j, 0, T_t, ALPHA, BETA, MU, T, w))
    for i in range(M):
        for j in range(M):
            C = np.append(C, del_L_beta(i, j, 0, T_t, ALPHA, BETA, MU, T, w))

    # return del Mu, del ALPHA, del BETA
    return np.append(np.append(A, B), C)


def second_derivative(T_t, ALPHA, BETA, MU, T, w):
    _, M = np.shape(ALPHA)
    # MATRIX IS LIKE :
    # ABC
    # DEF
    # GHI

    # Hessian are symmetric
    diag_matrices = np.zeros(M)
    for i in range(M):
        diag_matrices[i] = del_L_mu_mu(i, i, 0, T_t, ALPHA, BETA, MU, T, w)

    A = np.diag(diag_matrices)
    B = special_matrix_creator_rect(M, del_L_alpha_mu, del_L_alpha_mu_dif, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU, T=T,
                                    w=w)
    C = special_matrix_creator_rect(M, del_L_beta_mu, del_L_beta_mu_dif, T_t=T_t, ALPHA=ALPHA, BETA=BETA, MU=MU, T=T,
                                    w=w)
    D = np.transpose(B)
    E = special_matrix_creator_square(M, del_L_alpha_alpha, del_L_alpha_alpha_dif, del_L_alpha_alpha_dif_dif, T_t=T_t,
                                      ALPHA=ALPHA, BETA=BETA, MU=MU, T=T, w=w)
    F = special_matrix_creator_square(M, del_L_beta_alpha, del_L_beta_alpha_dif, del_L_beta_alpha_dif_dif, T_t=T_t,
                                      ALPHA=ALPHA, BETA=BETA, MU=MU, T=T, w=w)
    G = np.transpose(C)
    H = np.transpose(F)
    I = special_matrix_creator_square(M, del_L_beta_beta, del_L_beta_beta_dif, del_L_beta_beta_dif_dif, T_t=T_t,
                                      ALPHA=ALPHA, BETA=BETA, MU=MU, T=T, w=w)
    ans = np.zeros((M + 2 * M ** 2, M + 2 * M ** 2))
    lim1 = M - 1
    lim2 = M ** 2 + M - 1

    # print("A : ", A)
    # print("B : ", B)
    # print("C : ", C)
    # print("D : ", D)
    # print("E : ", E)
    # print("F : ", F)
    # print("G : ", G)
    # print("H : ", H)
    # print("I : ", I)
    ans[0:lim1 + 1, 0:lim1 + 1] = A
    ans[lim1 + 1:lim2 + 1, 0:lim1 + 1] = D
    ans[lim2 + 1:, 0:lim1 + 1] = G
    ans[0:lim1 + 1, lim1 + 1:lim2 + 1] = B
    ans[lim1 + 1:lim2 + 1, lim1 + 1:lim2 + 1] = E
    ans[lim2 + 1:, lim1 + 1:lim2 + 1] = H
    ans[0:lim1 + 1, lim2 + 1:] = C
    ans[lim1 + 1:lim2 + 1, lim2 + 1:] = F
    ans[lim2 + 1:, lim2 + 1:] = I
    # print("ans : ",ans)

    return ans