import sys
import recsys.prediction_algorithms
import recsys.dataset as data1
import pandas as pd
import recsys.evaluate
import surprise

# KNN = recsys.KNNBasic(40,1,{'name':'cosine','user_based':True})
# print KNN.predict()
headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

data = pd.read_csv('ml-100k/u.data', sep='\t', header=None, names=headers, index_col="MovieID")
print data.dtypes, '\n', '____________________'


#print type(data)
#print df
#print pd.melt(data_f,id_vars='MovieID')
#print (data.set_index('UserID').T)
#print (df.set_index('UserID').T)
print (data.rename(columns={'UserID': 'Users 1 ID'}).set_index(None).rename_axis().T)
