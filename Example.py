from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from surprise import Reader, Dataset, evaluate, print_perf, SVD, similarities, KNNBasic
import recsys
import numpy as np
import pandas as pd
import graphlab

# KNN = recsys.KNNBasic(40,1,{'name':'cosine','user_based':True})
# print KNN.predict()
headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

# TODO [DONE] Loading Dataset
df = pd.read_csv('ml-100k/u.data', sep='\t', names=('UserID', 'MovieID', 'Rating', 'TimeStamp'))
print df.dtypes, '\n', '____________________'


# TODO [DONE] Printing only selected columns
values = df.values
print values[:,0:2]

# TODO [DONE] Simple sorting of values
df_sorted_values = df.sort_values(['UserID', 'MovieID'])
print type(df_sorted_values)

# TODO [DONE] Printing Matrix of Dataset with NaN Values
print("*___________________*")
df_matrix = df.pivot_table(values='Rating', index='UserID', columns='MovieID')

# FOLL STATEMENTS PRODUCE THE SAME RESULT

# df3 = df.set_index(['UserID','MovieID'])['Rating'].unstack()
# df3 = df.pivot(index='UserID',columns='MovieID',values='Rating')
# df3 = df.groupby(['UserID','MovieID'])['Rating'].mean().unstack()
print df_matrix

reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1,5))
df = Dataset.load_from_file('ml-100k/u.data', reader=reader)
df.split(n_folds=5)

# TODO Similarity using KNN basic after applying Pearson & Cosine

ip = raw_input("Enter the choice of algorithm(1. Cosine/2. Pearson):")

if ip==1:
    one = 'cosine'
else:
    one = 'pearson'

cf = raw_input("Filtering Method: \n1.User based \n2.Item based \n Choice:")

if cf==1:
    two = True
else:
    two = False

sim_op = {'name':one,'user_based':two}
algo = KNNBasic(sim_options=sim_op)

perf = evaluate(algo,df,measures=['RMSE','MAE'])
print_perf(perf)

# TODO Generate Recommendations based on User/Item for its nearest neighbours

##algo.train('ml100k/u.data')
##testset =

# sim = similarities.cosine(10,10,22)
# print type(sim)
# predicted = cross_val_predict(X=df,y=None,cv=5)
# print predicted

# kf = KFold(n_splits=4)
# for i in kf.split(df):
#    print ("%d" % (i))
# TODO To produce recommendations

# TODO Perform K Fold CV on recommendations and print precision & recall

# END

