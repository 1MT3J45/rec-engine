import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)

train_data, test_data = cv.train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items)) # CREATING TRAIN DATA MATRIX from u.data
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items)) # CREATING TEST DATA MATRIX from u.data
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

item_prediction = predict(train_data_matrix, item_similarity, type='item')
print (item_prediction.shape)
user_prediction = predict(train_data_matrix, user_similarity, type='user')
print (user_prediction.shape)

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))
print 'Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix))

sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print 'The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%'

# ____________________________ TRIAL FOUR ___________________________________
#import pandas as pd
#import graphlab
#import time

#train_data = pd.read_pickle('train_data.pickle')
#test_data = pd.read_pickle('test_data.pickle')
#popularity_model = ('pop_model')

#algo = raw_input("COSINE | JACCARD | PEARSON \n Enter the name of algo:")

# Train Model
#item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type=algo)

# Make Recommendations:
#item_sim_recomm = item_sim_model.recommend(users=range(1, 6), k=5)
#item_sim_recomm.print_rows(num_rows=25)

#model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
#graphlab.show_comparison(model_performance, [popularity_model, item_sim_model])

#while (True):
#    pass

# _________________________ TRIAL THREE ___________________________
# f = open('mycsv.csv','rb')
# fo = open('out.csv','wb')

# go through each line of the file
# for line in f:
#    bits = line.split(',')
#    # change second column
#    bits[1] = '"NAME_UNDEFINED"'
#    # join it back together and write it out
#    fo.write( ','.join(bits) )



# f.close()
# fo.close()
# x=open("out.csv")
# y=x.read()
#print (y)

# _______________________TRIAL TWO_____________________
#normalize = lambda x: "%.4f" % float(X)

# df1 = pd.read_csv("ml-100k/u.item", index_col=(0, 1), usecols=(1,2), header=None, converters=dict.fromkeys([1, 2], normalize))
# df2 = pd.read_csv("ml-100k/u.item", index_col=(0, 1), usecols=(1,2), header=None, converters=dict.fromkeys([1, 2], normalize))

# result = df1.join(df2, how='inner')
# result.to_csv("output.csv", header=None)
# _______________________TRIAL ONE_____________________
# Reading users file:
# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# try:
#    reader = pd.read_csv('ml-100k/u.item', delimiter='|', usecols=[0, 1, 2, 4], nrows=3)
#    print ("\n")
#    print(reader)
#    print (range(5))
#    # col = int(input("Input number column to print"))
#    row = int(input("Input number of rows to print"))
#    list1 = [0, 0, 0, 0]
#    for i in range(3):
#        list1[i] = input("Insert val")
#        print (i)
#        i += 1
#    print (pd.read_csv('ml-100k/u.item', delimiter='|', usecols=list1, nrows=row))
#    # match_list = pd.read_csv(pd.read_csv('ml-100k/u.item', delimiter='|', usecols=list1, nrows=row))
#    # print ("\n THIS IS NEW MATCH LIST", match_list)

#    rec_list=()
#    records=int(input("How many records to collect?\n"))
#    for i in range(records):
#        abc = (pd.read_csv(pd.read_csv('ml-100k/u.item', delimiter='|', usecols=[0], nrows=records)))
#        i +=1
#    # rec_list = ("Sr",1,2,3)
#        print (abc)
# finally:
#    exit(0)
