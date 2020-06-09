from class_estimator import *

est = Estimator(pd.DataFrame(columns=['variable', 'n', 'm',
                                  'time estimation', 'weight function',
                                  'value', 'T_max', 'true value']))

def f(estimator):
    estimator.DF = (estimator.DF).append(pd.DataFrame(
            {"time estimation": [1],
             "variable": ["nu"],
             "n": [2],
             "m": [0],
             "weight function": [2],
             "value": [2],
             'T_max': [3],
             'true value': [4]
             }), sort=True
        )

d = pd.DataFrame(columns=['variable'])
print(d)
d.append(pd.DataFrame({'variable': [3]}))
print(d)
