import sys
import recsys.prediction_algorithms
import recsys.dataset as data1
import pandas as pd
import csv
import surprise

# KNN = recsys.KNNBasic(40,1,{'name':'cosine','user_based':True})
# print KNN.predict()
headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

df = pd.read_csv('ml-100k/u.data', sep='\t', names=('UserID', 'MovieID', 'Rating', 'TimeStamp'))
print df.dtypes, '\n', '____________________'
print df

df2 = df.sort_values(['UserID','MovieID'])
print type(df2)
df2.to_csv("sorted.csv", sep="\t")

print("*___________________*")
df3 = df.pivot_table(values='Rating',index='UserID',columns='MovieID', aggfunc='mean')
print df3
df3.to_csv("matrix.csv",sep="\t")
#print type(data)
#print df
#print pd.melt(data_f,id_vars='MovieID')
#print (data.set_index('UserID').T)
#print (df.set_index('UserID').T)
#print (df.rename(columns={'UserID': 'Users 1 ID'}).set_index(None).rename_axis().T)
