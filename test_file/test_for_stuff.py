import pandas as pd

path = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\second_estimation\super_smaller_4_second.csv'
df = pd.read_csv(path)#,
            #index_col='Employee',
            #parse_dates=['Hired'],
            #header=0,
            #names=['Employee', 'Hired','Salary', 'Sick Days'])
print(df)

for i in range( len(df.index) ):
    # print(df.at[i, 'weight function'])
    df.at[i, 'weight function'] = "Adaptive Biweight with first width 975.0"
print(df)

df.to_csv(path)
