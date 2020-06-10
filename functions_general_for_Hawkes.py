# create the exact good list of size M*M*3
# the function returns [   [[0]], [[0]], [[0]]   ]
def multi_list_generator(size):
    if size < 1:
        raise ("Size not big enough.")
    ans = []
    interior_list = [0] * size
    for j in range(3):
        medium_list = []
        for i in range(size):
            medium_list.append(interior_list.copy())  # not recreated every i so need copy.
            if j == 0 and i == 0:  # case for mu, reduces space used.
                medium_list = medium_list[0]
                break
        ans.append(medium_list)  # no copy because recreated every j
    return ans

#BIANCA-HERE if we create a ESTIM-HP then we can put that function inside of the class.
def mean_HP(estimator, separator = None):
    ## separators is a list, of the estimators to gather together.
    separators = ['variable', 'm', 'n']
    if separator is not None:
        for strg in separator: separators.append(strg)
    dict_of_means = estimator.DF.groupby(separators)['value'].mean()
    key_parameters = ['nu', 'alpha', 'beta']
    ans_N, ans_A, ans_B = [], [], []
    M = estimator.DF["m"].max() + 1
    for i in range(M):
        ans_N.append(dict_of_means[('nu', i, 0)])
        for j in range(M):
            if not j :
                ans_A.append([])
                ans_B.append([])
            ans_A[i].append(dict_of_means[('alpha', i, j)])
            ans_B[i].append(dict_of_means[('beta', i, j)])

    return [ans_N, ans_A, ans_B]