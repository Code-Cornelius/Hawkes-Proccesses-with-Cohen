##### normal libraries


##### my libraries

##### other files
from classes.class_kernel import *


class Estimator_Hawkes(Estimator):
    # DF is a dataframe from pandas. Storing information inside is quite easy, easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self):
        super().__init__(pd.DataFrame(columns=['variable', 'n', 'm',
                                               'time estimation', 'weight function',
                                               'value', 'T_max', 'true value', 'number of guesses']))

    def mean(self, separator=None):
        ## separators is a list, of the estimators to gather together.
        separators = ['variable', 'm', 'n']
        if separator is not None:
            for str in separator: separators.append(str)
        dict_of_means = self.DF.groupby(separators)['value'].mean()
        ans_N, ans_A, ans_B = [], [], []
        M = self.DF["m"].max() + 1
        for i in range(M):
            ans_N.append(dict_of_means[('nu', i, 0)])
            for j in range(M):
                if not j:
                    ans_A.append([])
                    ans_B.append([])
                ans_A[i].append(dict_of_means[('alpha', i, j)])
                ans_B[i].append(dict_of_means[('beta', i, j)])

        return [ans_N, ans_A, ans_B]

# example:
#
#  estimators = estimators.append(pd.DataFrame(
#                             {"time estimation": T[i],
#                              "variable": "alpha",
#                              "n": s,
#                              "m": t,
#                              "weight function": str(function_weight[i_weights].name),
#                              "value": ALPHA_HAT[s, t]
#                              }), sort=True
#                         )
#
# estimators = estimators.append(pd.DataFrame(
#     {"time estimation": T[i],
#      "variable": "nu",
#      "n": s,
#      "m": 0,
#      "weight function": str(function_weight[i_weights].name),
#      "value": MU_HAT[s]
#      }), sort=True
# )
