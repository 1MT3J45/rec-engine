
import pandas as pd
import graphlab
import time
import pickle

# pass in column names for each CSV and read them using pandas.
# Column names available in the readme file

# Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

# Reading items file:
i_cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

print(users.shape)  # Printing Rows x Cols for USERS
users.head()        # Not Printing only Headers than entire data

print(ratings.shape) # Printing Rows x Cols for Ratings
ratings.head()       # Not Printing only Headers than entire data

print(items.shape) # Printing Rows x Cols for ITEMS
items.head()       # Not Printing only Headers than entire data


# Fetching DB of Ratings per Movie for every user
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/u2.base', sep='\t', names=r_cols, encoding='latin-1')

# Fetching DB of Testing Data for Movies Users have Rated
ratings_test = pd.read_csv('ml-100k/u2.test', sep='\t', names=r_cols, encoding='latin-1')
info = pd.read_csv('ml-100k/u.info', sep='|')

print ("________ RATING SIZE __________")
print(ratings_base.shape)

print ("________ RATING BASE __________")
print(ratings_base)

print ("________ RATING TEST __________")
print(ratings_test)

print ("________ GEN. INFO __________")
print(info)

train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

print ("________ TRAIN DATA __________")
print (train_data)

train_data.to_dataframe().to_pickle(r'train_data.pickle')
test_data.to_dataframe().to_pickle(r'test_data.pickle')
# ____________ DEFINING POPULARITY MODEL _______________

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')
print ("[|||||||POPULARITY MODEL TYPE >",type(popularity_model)," < ]")
popularity_model.save('pop_model')

# Get recommendations for first 5 users and print them
# users = range(1,6) specifies user ID of first 5 users
# k=5 specifies top 5 recommendations to be given
print ("______ POPULARITY MODEL ________")
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

print ("________ RATINGS BASE __________")
print(ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20))



# Preparing a USER Similarity model
from math import sqrt


def sums(sum_of_euclidean_distance):
    pass

ml_dataset = pd.read_csv('ml-100k/u.data')
dataset = pd.read_csv('ml-100k/u.data',)

def similarity_score(person1, person2):
    # This Returns the ration euclidean distance score of person 1 and 2

    # To get both rated items by person 1 and 2
    both_viewed = {}

    for item in dataset[person1]:
        if item in dataset[person2]:
            both_viewed[item] = 1

        # The Conditions to check if they both have common rating items
        if len(both_viewed) == 0:
            return 0

        # Finding Euclidean distance
        sum_of_euclidean_distance = []

        for item in dataset[person1]:
            if item in dataset[person2]:
                sum_of_euclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item], 2))
        sum_of_euclidean_distance = sums(sum_of_euclidean_distance)

        return 1 / (1 + sqrt(sum_of_euclidean_distance))


def pearson_correlation(person1, person2):
    # To get both rated items
    both_rated = {}
    for item in dataset[person1]:
        if item in dataset[person2]:
            both_rated[item] = 1

    number_of_ratings = len(both_rated)

    # Checking for ratings in common
    if number_of_ratings == 0:
        return 0

    # Add up all the preferences of each user
    person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
    person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

    # Sum up the squares of preferences of each user
    person1_square_preferences_sum = sum([pow(dataset[person1][item], 2) for item in both_rated])
    person2_square_preferences_sum = sum([pow(dataset[person2][item], 2) for item in both_rated])

    # Sum up the product value of both preferences for each item
    product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

    # Calculate the pearson score
    numerator_value = product_sum_of_both_users - (
    person1_preferences_sum * person2_preferences_sum / number_of_ratings)
    denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum, 2) / number_of_ratings) * (
    person2_square_preferences_sum - pow(person2_preferences_sum, 2) / number_of_ratings))

    if denominator_value == 0:
        return 0
    else:
        r = numerator_value / denominator_value
        return r


def most_similar_users(person, number_of_users):
    # returns the number_of_users (similar persons) for a given specific person
    scores = [(pearson_correlation(person, other_person), other_person) for other_person in dataset if
              other_person != person]

    # Sort the similar persons so the highest scores person will appear at the first
    scores.sort()
    scores.reverse()
    return scores[0:number_of_users]


def user_recommendations(person):
    # Gets recommendations for a person by using a weighted average of every other user's rankings
    totals = {}
    simSums = {}
    rankings_list = []
    for other in dataset:
        # don't compare me to myself
        if other == person:
            continue
        sim = pearson_correlation(person, other)
        print "UBCF Similarity>>>", sim

        # ignore scores of zero or lower
        if sim <= 0:
            continue
        for item in dataset[other]:

            # only score movies i haven't seen yet
            if item not in dataset[person] or dataset[person][item] == 0:
                # Similarity * score
                totals.setdefault(item, 0)
                totals[item] += dataset[other][item] * sim
                # sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim
                # print dataset[item]
                # Create the normalized list

    rankings = [(total / simSums[item], item) for item, total in totals.items()]
    rankings.sort()
    rankings.reverse()
    # returns the recommended items
    recommendataions_list = [recommend_item for score, recommend_item in rankings]
    return recommendataions_list


io = raw_input("Enter User ID: ")
print user_recommendations(io)


# _____________________ TRIAL FIVE _______________________________
# import numpy as np
# import pandas as pd
# from sklearn import cross_validation as cv
# from sklearn.metrics.pairwise import pairwise_distances
# from sklearn.metrics import mean_squared_error
# from math import sqrt

'''
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
'''

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
