from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from surprise import Reader, Dataset, evaluate, print_perf, SVD
import recsys
import numpy as np
import pandas as pd

# KNN = recsys.KNNBasic(40,1,{'name':'cosine','user_based':True})
# print KNN.predict()
headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

df = pd.read_csv('ml-100k/u.data', sep='\t', names=('UserID', 'MovieID', 'Rating', 'TimeStamp'))
print df.dtypes, '\n', '____________________'


# Printing only selected columns
values = df.values
print values[:,0:2]

# Simple sorting of values
df2 = df.sort_values(['UserID','MovieID'])
print type(df2)

print("*___________________*")
df3 = df.pivot_table(values='Rating',index='UserID',columns='MovieID')
## FOLL STATEMENTS PRODUCE THE SAME RESULT
# df3 = df.set_index(['UserID','MovieID'])['Rating'].unstack()
# df3 = df.pivot(index='UserID',columns='MovieID',values='Rating')
# df3 = df.groupby(['UserID','MovieID'])['Rating'].mean().unstack()
print df3

reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1,5))
df = Dataset.load_from_file('ml-100k/u.data', reader=reader)
df.split(n_folds=5)

perf = evaluate(SVD(),df,measures=['RMSE','MAE'])
print_perf(perf)

#predicted = cross_val_predict(X=df,y=None,cv=5)
#print predicted

# kf = KFold(n_splits=4)
# for i in kf.split(df):
#    print ("%d" % (i))

