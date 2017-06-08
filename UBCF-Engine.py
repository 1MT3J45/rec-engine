import pandas as pd
import graphlab
import numpy as np
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

