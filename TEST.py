# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import csv
#
# # TODO -- Initiate Selection of Matrix Files
#
# file=0
# try:
#     file = input("Choose your File Name:\n 0. MainMatrix_IBCF.csv\n 1. MainMatrix_UBCF.csv\n ch no (0 or 1):")
# except Exception as e:
#     print "NOTE:",e
#
# dataset = pd.DataFrame
# if file is 0:
#     dataset = pd.read_csv("MainMatrix_IBCF.csv")
# elif file is 1:
#     dataset = pd.read_csv("MainMatrix_UBCF.csv")
#
#
# def graphCall1(wcss):
#     plt.plot(range(1, 7), wcss)
#     plt.title('The Elbow Method')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('WCSS')
#     plt.show()
#
#
# def graphCall2(X, y_kmeans, kmeans):
#     # Visualising the clusters
#     plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
#     plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
#     plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
#     plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
#     # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
#     # In case of 5th Cluster arises
#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
#
#     plt.title('Clusters of Users')
#     plt.xlabel('Movies')
#     plt.ylabel('Rating')
#     plt.legend()
#     plt.show()
#
# def CookClusters(dataset):
#     dataset = dataset.dropna()
#
#     # TODO -- Convert it to NDarrays for Clustering using KMeans
#
#     X = dataset.values
#
#     # TODO -- Calculate WCSS vs No. Of Clusters to find Optimal no. of Clusters
#
#     wcss = []
#     for i in range(1, 7):
#         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)
#     graphCall1(wcss)
#
#     # TODO -- Predict Optimal no. of Clusters for each record
#     # KMeans Clustering
#     no_of_clusters = 4
#     kmeans = KMeans(n_clusters=no_of_clusters)    # KMeans clustering
#     y_kmeans = kmeans.fit_predict(X)                                                # Predicted Clusters for Each record
#
#     # Conversion into List from Dataframe
#     y_km = list(y_kmeans)       # Converting ndarray into List
#     X_df = pd.DataFrame(X)      # Converting ndarray into Dataframe
#
#     # TODO -- Separate out the Clusters into Dataframe based on Prediction and Visualize
#
#     cluster = {k:g for k, g in X_df.groupby(y_kmeans)}  # Dividing records into Clusters
#     # X_df = X_df.dropna(axis=0)
#
#     # D0 = cluster[0]
#     # D1 = cluster[1]
#     # D2 = cluster[2]
#     # D3 = cluster[3]
#
#     graphCall2(X, y_kmeans, kmeans)
#
#     return cluster
#
# # TODO -- CALL COOK CLUSTERS and Make Cluster Dataframes
# cooked = CookClusters(dataset=dataset)
#
# D0 = cooked[0]
# D1 = cooked[1]
# D2 = cooked[2]
# D3 = cooked[3]
#
# # TODO -- Take Randomized Input of Users >= No. of Clusters
#
# ip_df = pd.read_csv("input.csv")
# count_set = set(ip_df.iloc[0:,0])
# if len(count_set) > 4:
#     print "Input File has Less no. of Users than Number of Clusters\nExiting"
#     exit(0)
# else:
#     pass
#
# CookedInput = CookClusters(dataset=ip_df)
#
#
# def recipe_2(dataset):
#     dataset = dataset.dropna()
#
#     # TODO -- Convert it to NDarrays for Clustering using KMeans
#
#     X = dataset.values
#
#     # TODO -- Calculate WCSS vs No. Of Clusters to find Optimal no. of Clusters
#
#     wcss = []
#     for i in range(1, 7):
#         kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
#         kmeans.fit(X)
#         wcss.append(kmeans.inertia_)
#     graphCall1(wcss)
#
#     # TODO -- Predict Optimal no. of Clusters for each record
#     # KMeans Clustering
#     no_of_clusters = 4
#     kmeans = KMeans(n_clusters=no_of_clusters, init='k-means++', random_state=0)    # KMeans clustering
#     y_kmeans = kmeans.fit_predict(X)                                                # Predicted Clusters for Each record
#
#     # Conversion into List from Dataframe
#     y_km = list(y_kmeans)       # Converting ndarray into List
#     X_df = pd.DataFrame(X)      # Converting ndarray into Dataframe
#
#     # TODO -- Separate out the Clusters into Dataframe based on Prediction and Visualize
#
#     cluster = {k:g for k, g in X_df.groupby(y_kmeans)}  # Dividing records into Clusters
#     # X_df = X_df.dropna(axis=0)
#
#     # D0 = cluster[0]
#     # D1 = cluster[1]
#     # D2 = cluster[2]
#     # D3 = cluster[3]
#
#     graphCall2(X, y_kmeans, kmeans)
#
#     return cluster

# -- TODO ------------------------------ KITCHEN 2

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ch = int(input("Select Appropriate Matrix\n 1. MainMatrix_IBCF.csv\n 2. MainMatrix_UBCF\n Choice:"))

if ch is 1:
    data_set = pd.read_csv('MainMatrix_IBCF.csv')
    data_set = data_set.dropna()
elif ch is 2:
    data_set = pd.read_csv('MainMatrix_UBCF.csv')
    data_set = data_set.dropna()
else:
    data_set = pd.DataFrame

def recipe_one(dataset):
    dataset = dataset.dropna()

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(dataset)

    labels = kmeans.labels_
    pd.set_option('display.max_rows', dataset.describe().shape[1])

    C0 = dataset[labels == 0]
    C1 = dataset[labels == 1]
    C2 = dataset[labels == 2]
    C3 = dataset[labels == 3]
    C4 = dataset[labels == 4]

    li = [C0, C1, C2, C3, C4]

    return li

Fresh_Clusters = recipe_one(dataset=data_set)

fc0 = Fresh_Clusters[0]
fc1 = Fresh_Clusters[1]
fc2 = Fresh_Clusters[2]
fc3 = Fresh_Clusters[3]
fc4 = Fresh_Clusters[4]

input_read = pd.read_csv('input.csv')
input_df = input_read.pivot(index='uid', columns='iid', values='rat')

try:
    young_Clusters = recipe_one(dataset=input_df)

except Exception as e:
    print "NOTE", e

yc0 = young_Clusters[0]
yc1 = young_Clusters[1]
yc2 = young_Clusters[2]
yc3 = young_Clusters[3]
yc4 = young_Clusters[4]
