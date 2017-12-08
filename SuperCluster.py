import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

file=0
try:
    file = input("Choose your File Name:\n 0. MainMatrix_IBCF.csv\n 1. MainMatrix_UBCF.csv\n ch no (0 or 1):")
except Exception as e:
    print "NOTE:",e

dataset = pd.DataFrame
if file is 0:
    dataset = pd.read_csv("MainMatrix_IBCF.csv")
elif file is 1:
    dataset = pd.read_csv("MainMatrix_UBCF.csv")
elif file is 2:                                 # Experimental
    dataset = pd.read_csv("input.csv")
    dataset.dropna()

dataset = dataset.drop_duplicates()
dataset = dataset.dropna(axis=1)
print dataset.shape
X = dataset.iloc[:,1:1682].values               # Experimental
X = np.delete(X, (941),axis=0)                  # Experimental

wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# KMeans Clustering
no_of_clusters = 4
kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', random_state=0)     # KMeans clustering
y_kmeans = kmeans.fit_predict(X)        # Predicted Clusters for Each record

# Conversion into List from Dataframe
y_km = list(y_kmeans)       # Converting ndarray into List
X_df = pd.DataFrame(X)
X_df = pd.DataFrame(X_df, columns=np.arange(1,X_df.columns.__len__()), index=np.arange(1,X_df.__len__()+1))

cluster = {k:g for k, g in X_df.groupby(y_kmeans)}  # Dividing records into Clusters
X_df = X_df.dropna(axis=0)

D0 = cluster[0]
D1 = cluster[1]
D2 = cluster[2]
D3 = cluster[3]

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    #In case of 5th Cluster arises
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Users')
plt.xlabel('Movies')
plt.ylabel('Rating')
plt.legend()
plt.show()

# TODO --- Read a Randomized input file
ip_df = pd.read_csv("input.csv")

def ip_KMcluster(ip_df):
    dataset = ip_df
    dataset = dataset.drop_duplicates()
    dataset = dataset.dropna()
    tup = dataset.shape
    print "Total Size of Input Data\n", tup
    X = dataset.iloc[:, 0:tup[1]].values  # Experimental

    wcss = []
    for i in range(1, 7):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 7), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # KMeans Clustering
    no_of_clusters = 4
    kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', random_state=0)  # KMeans clustering
    y_kmeans = kmeans.fit_predict(X)  # Predicted Clusters for Each record

    # Conversion into List from Dataframe
    y_km = list(y_kmeans)  # Converting ndarray into List
    X_df = pd.DataFrame(X)
    X_df = pd.DataFrame(X_df, columns=np.arange(1, X_df.columns.__len__()))

    cluster = {k: g for k, g in X_df.groupby(y_km)}  # Dividing records into Clusters
    X_df = X_df.dropna(axis=0)

    D0 = cluster[0]
    D1 = cluster[1]
    D2 = cluster[2]
    D3 = cluster[3]

    li = [D0, D1, D2, D3]; print "\nSize of Clusters:\n"
    for i in range(len(li)):
        print "D%d :"%(i) , li[i].shape

    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
    # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    # In case of 5th Cluster arises
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

    plt.title('Clusters of Users')
    plt.xlabel('Movies')
    plt.ylabel('Rating')
    plt.legend()
    plt.show()

    return cluster

ip_cluster = ip_KMcluster(ip_df)

def merge_Clusters(ip_cluster, mega_cluster):

    # '''
    # ip_cluster is a Dictionary containing input file's cluster
    # mega_cluster is a Disctionary containing actual clusters
    # Conversion of ndarray to Dataframe is necessary for prediction
    # '''

    ip_df0 = pd.DataFrame(ip_cluster[0])
    ip_df1 = pd.DataFrame(ip_cluster[1])
    ip_df2 = pd.DataFrame(ip_cluster[2])
    ip_df3 = pd.DataFrame(ip_cluster[3])

    # '''Getting Mega Cluster
    # and converting it to Dataframe'''
    df0 = pd.DataFrame(mega_cluster[0])
    df1 = pd.DataFrame(mega_cluster[1])
    df2 = pd.DataFrame(mega_cluster[2])
    df3 = pd.DataFrame(mega_cluster[3])

    # '''Merging the Dataframes based on cluster Results'''
    df0 = pd.DataFrame.append(ip_df0)
    df1 = pd.DataFrame.append(ip_df1)
    df2 = pd.DataFrame.append(ip_df2)
    df3 = pd.DataFrame.append(ip_df3)

# TODO --- Push them to respective cluster


# TODO --- Generate Predictions
from surprise import Reader, Dataset, KNNBasic
from collections import defaultdict

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    uid = None

    # MAPPING PREDICTIONS TO EACH USER
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # THEN SORT THE PREDICTIONS FOR EACH USER AND RETRIEVE THE K Highest ones

    for iid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# ----------------------------------------- GENERATION OF PREDICTION MATRIX
def gen_pred_matrix(co_pe):

    # INITIALIZE REQUIRED PARAMETERS
    # path = 'ml-100k/u.user'
    sim_op = {'name': co_pe, 'user_based': file}
    algo = KNNBasic(sim_options=sim_op)
    global ip_df
    ip_df = ip_df.iloc[0:,:5]

    reader = Reader(line_format="uid 1 2 3 4 5", sep=',', rating_scale=(1, 5))
    df = Dataset.load_from_file(ip_df, reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    algo.train(trainset)
    print "ALGORITHM USED", co_pe

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset=testset)

    top_n = get_top_n(predictions, 5)

    # ---------------------------------------------------- UBCF as is

    csvfile = 'PRED-MATRIX-CLUSTER %d.csv'%(np.random.randint(100,200))
    with open(csvfile, "w") as output:
        writer = csv.writer(output, delimiter=',', lineterminator='\n')
        writer.writerow(['uid']+range(0,len(ip_df)))
        for uid, user_ratings in top_n.items():
            for (iid, r) in user_ratings:
                value = uid, iid, r
                writer.writerow(value)
    print "Done! Check File with Name:",csvfile


# TODO --- Predict the NaN values for new Inputs

# TODO --- Predict the rest of the ratings of those new users

# Control Flow
ch = input("Choice of Algorithm?\n 1. Cosine\n 2. Pearson")

if ch is 1:
    algorithm = 'cosine'
elif ch is 2:
    algorithm = 'pearson'

# gen_pred_matrix(algorithm)