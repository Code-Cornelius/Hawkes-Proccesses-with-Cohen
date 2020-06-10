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
    intermediate_DF = estimator.DF.groupby(separators)
    print("I \n", intermediate_DF)
    print( "II \n",
        intermediate_DF['value'].mean()
    )
    # la dernière expression recupere une série. Juste à mettre la clé genre 'alpha', 0, 0
    return