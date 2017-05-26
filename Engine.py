import pandas as pd
import graphlab
import csv

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
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')

# Fetching DB of Testing Data for Movies Users have Rated
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
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

# ____________ DEFINING POPULARITY MODEL _______________

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')

# Get recommendations for first 5 users and print them
# users = range(1,6) specifies user ID of first 5 users
# k=5 specifies top 5 recommendations to be given
print ("______ POPULARITY MODEL ________")
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

print ("________ RATINGS BASE __________")
print(ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20))

# Preparing an ITEM Similarity model

# Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')
# Make Recommendations
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=20)

# <editor-fold desc="Printing Recommendation with Movie names but not limited to Rank">
# df_item_sim = item_sim_recomm.to_dataframe()        # DF2 Recom | user_id | movie_id | score | rank |
# df_movies = pd.read_csv('ml-100k/u.item',sep='|')   # DF3 Items | ID | MOVIE NAME (YEAR)| REL.DATE | NULL | IMDB LINK|

# df_item_sim['movie_id'] = df_item_sim['movie_id'].replace(df_movies.set_index('ID')['MOVIE NAME (YEAR)'])
# print ("________ REC W/ MOVIE NAMES ___________")
# print (df_item_sim)
# </editor-fold>

# print ("***********_______TO BE USED LATER________************")
# df1 = train_data.to_dataframe()
# df1.to_csv("train_data.csv", sep=",")

# print ("________DF1-PRINTED________")
# print (df1) # .set_index("ID")['MOVIE NAME (YEAR)'])

# df2=df1                                       # DF2 is TRAINING DATA with  |user_id | movie_id | rating | unix_timestamp|
# df3= pd.read_csv('ml-100k/u.item',sep='|')    # DF3 is ITEM DATA with ID|MOVIE NAME (YEAR)|REL.DATE|NULL|IMDB LINK|

# df2['movie_name'] = df2['movie_id'].replace(df3.set_index('ID')['MOVIE NAME (YEAR)'])

#print ("________ DF2-PRINTED________")
# print (df2)

