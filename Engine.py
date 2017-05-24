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

print(users.shape)
users.head()

print(ratings.shape)
ratings.head()

print(items.shape)
items.head()

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
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

df1 = train_data.to_dataframe()
df1.to_csv("train_data.csv", sep=",")

#with open('train_data.csv','wb') as myfile:
#    wr = csv.writer(myfile)
#    wr.writerow(train_data)
#myfile.close()
#-----------------------REPLACING MOVIE ID WITH MOVIE NAMES-----#

f = open('ml-100k/u.item', 'rw')
fo = open('train_data.csv', 'rw')

# go through each line of the file
#for line in f:
#    bits = line.split('|')
    # change second column
#    bits[1] = 'name'
    # join it back together and write it out
#    fo.write(','.join(bits))
#print (f.read())

f.close()
fo.close()
#x = open("out.csv")
#y = x.read()
#print (y)