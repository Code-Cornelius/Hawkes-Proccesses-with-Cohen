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
        # the output format is list of lists with on each line [ans_N, ans_A, ans_B],
        # and on every single additional dimension, the separator.
        ## separators is a list, of the estimators to gather together.
        separators = ['variable', 'm', 'n']
        M = self.DF["m"].max() + 1
        ans_dict = {}

        # if separator is not None:
        #     for str in separator: separators.append(str)


        global_dict, keys = self.groupby_DF(separator)
        for key in keys:
            data = global_dict.get_group(key)
            dict_of_means = data.groupby(separators)['value'].mean()
            ans_N, ans_A, ans_B = [], [], []

            for i in range(M):
                ans_N.append(dict_of_means[('nu', i, 0)])
                for j in range(M):
                    if not j: # if j == 0
                        ans_A.append([])
                        ans_B.append([])
                    # we append to this new small list the j's.
                    ans_A[i].append(dict_of_means[('alpha', i, j)])
                    ans_B[i].append(dict_of_means[('beta', i, j)])
            # i get triple list like usually.
            ans_dict[ key ] = [ans_N, ans_A, ans_B]
        return ans_dict

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
