import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

dataset = pd.read_csv("universal_ibcf.csv")
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()
print dataset.shape
X = dataset.iloc[:,1:1682].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# KMeans Clustering
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)     # KMeans clustering
y_kmeans = kmeans.fit_predict(X)        # Predicted Clusters for Each record

# Conversion into List from Dataframe
y_km = list(y_kmeans)       # Converting ndarray into List
X_df = pd.DataFrame(X, index=np.arange(1,X.__len__()+1))      # Converting ndarray into Dataframe
X_df = pd.DataFrame(columns=X_df.columns.__len__()+1)

cluster = {k:g for k, g in X_df.groupby(y_kmeans)}  # Dividing records into Clusters

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
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Users')
plt.xlabel('Rating')
plt.ylabel('Movies')
plt.legend()
plt.show()

# TODO --- Take a Randomized input file

# TODO --- Predict the cluster for them

# TODO --- Push them to respective cluster

# TODO --- Predict the rest of the ratings of those new users