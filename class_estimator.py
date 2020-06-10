import numpy as np
import pandas as pd


class Estimator:
    # DF is a dataframe from pandas. Storing information inside is quite easy, easily printable and easy to collect back.
    # once initialize, one can add values. Each row is one estimator
    def __init__(self, DF):
        self.DF = DF

    def __repr__(self):
        return repr(self.DF)

    def function_upon_separeted_data(self, separator, fct, name, **kwargs):
        # separator is a string
        # fct is a fct
        # name is the name of a column.
        # one value is one parameter... is it enough parameter ?
        # the function does create a new column in the DF, by looking at the data in the separator and applying the function to it.
        self.DF[name] = self.DF.apply(lambda row: fct(row[separator], **kwargs), axis=1)
        return

    # TODO computation about the estimators. Like compute the mean stuff like that, and return the value.


# it corresponds to S^2. This is the empirical estimator of the variance.
def estimateur_variance(x_real, mean):
    # x real is a vector numpy array
    # mean a constant
    ans = 0
    for i in range(len(x_real)):
        ans += x_real[i] * x_real[i]
    ans *= 1 / len(x_real)
    ans -= mean * mean
    return ans



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