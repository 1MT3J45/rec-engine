import pandas as pd
import graphlab
import time

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

print users.shape
users.head()

print ratings.shape
ratings.head()

print items.shape
items.head()

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/u5.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/u5.test', sep='\t', names=r_cols, encoding='latin-1')
print (ratings_base.shape)
print (ratings_test.shape)

train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id',
                                                          target='rating')

# Get recommendations for first 5 users and print them
# users = range(1,6) specifies user ID of first 5 users
# k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1, 6), k=5)
popularity_recomm.print_rows(num_rows=25)

ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)

# Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id',
                                                             target='rating', similarity_type='cosine')

# Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1, 6), k=5)
item_sim_recomm.print_rows(num_rows=25)

model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance, [popularity_model, item_sim_model])

while (True):
    pass

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
