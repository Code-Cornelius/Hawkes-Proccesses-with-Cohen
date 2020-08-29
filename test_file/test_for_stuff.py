import pandas as pd

path = r'C:\Users\nie_k\Desktop\travail\RESEARCH\RESEARCH COHEN\Hawkes process Work\csv_files\first_estimations\super_0_first.csv'
df = pd.read_csv(path)#,
            #index_col='Employee',
            #parse_dates=['Hired'],
            #header=0,
            #names=['Employee', 'Hired','Salary', 'Sick Days'])


print( df.iloc[1000]['time estimation'])
df['time estimation'] = df['time estimation'].round(decimals= 6)
print( df.iloc[1000]['time estimation'])


df.to_csv(path, index = False)
