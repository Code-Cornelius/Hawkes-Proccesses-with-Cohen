from classes.class_Estimator_Hawkes import *
import time
import pandas as pd


df = pd.read_csv(r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\estimators_bianca.csv')#,
            #index_col='Employee',
            #parse_dates=['Hired'],
            #header=0,
            #names=['Employee', 'Hired','Salary', 'Sick Days'])
print(df)

for i in range( len(df.index) ):
    print(df.at[i, 'weight function'])
    df.at[i, 'weight function'] = "Biweight 5040 width"
print(df)

df.to_csv('super_0_first.csv')
