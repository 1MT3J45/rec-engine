import pandas as pd
import graphlab
import time

train_data = pd.read_pickle('train_data.pickle')
test_data = pd.read_pickle('test_data.pickle')
popularity_model = ('pop_model')

algo = raw_input("COSINE | JACCARD | PEARSON \n Enter the name of algo:")

# Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type=algo)

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
