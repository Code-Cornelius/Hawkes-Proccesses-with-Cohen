from class_Graph_Estimator_Hawkes import *
import time

M = 1000
my_list = [[] for i in range(M)]
for i in range(M):
    for j in range(M):
        my_list[i].append(0)
my_numpy_list = [ np.full(M,1) for i in range(M) ]
time1 = time.time()
for j in range(1000):
    for i in range(10000):
        my_list[0][0]
print( "1  ", time.time() - time1)

time1 = time.time()
for j in range(1000):
    test_list = my_list[0]
    for i in range(10000):
        test_list[0]
print("2 ",time.time() - time1)

time1 = time.time()
for j in range(1000):
    for i in range(10000):
        my_numpy_list[0][0]
print("3 ", time.time() - time1)

time1 = time.time()
for j in range(1000):
    my_numpy_test_list = my_numpy_list[0]
    for i in range(10000):
        my_numpy_test_list[0]
print( "4  ", time.time() - time1)


time.sleep(100
           )
M = 500
dict = {}
my_list = [[[ 1 ]*M]*M]*M
for k1 in range(M):
    print(k1)
    for k2 in range(M):
        for k3 in range(M):
            dict[(k1,k2,k3)] = 1

print("done creating")
my_time = time.time()
for j in range(20000):
    my_list[5][5][5]
print("1) : ",time.time() - my_time )


my_time = time.time()
for j in range(20000):
    dict[(k1,k2,k3)]
print("2) : ", time.time() - my_time )


list1 = [ 1 for i in range(M)]
list2 = np.zeros(M)
my_time = time.time()
for j in range(20000):
    list1[10]
print("3) : ",time.time() - my_time )


my_time = time.time()
for j in range(20000):
    list2[10]
print("4) : ", time.time() - my_time )

time.sleep(100)



est = Estimator_Hawkes()

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
