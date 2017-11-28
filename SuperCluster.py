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
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)
value = 0

# X_df = pd.DataFrame(X)
#
# # Adding individual values into Cluster files
# clf0 = pd.DataFrame
# clf1 = pd.DataFrame
# clf2 = pd.DataFrame
# clf3 = pd.DataFrame
# _0, _1, _2, _3 = 0,0,0,0
# for i in range(len(X)):
#     if y_kmeans[i] is 0:
#         clf0.append(X_df[i])
#         _0+=1
#     elif y_kmeans[3] is 1:
#         clf1.loc[_1] = pd.concat(X_df[i])
#         _1 += 1
#     elif y_kmeans[i] is 2:
#         clf1.loc[_2] = X_df[i]
#         _2 += 1
#     elif y_kmeans[i] is 3:
#         clf1.loc[_3] = X_df[i]
#         _3 += 1


csvfile = 'super_clusterX.csv'
with open(csvfile, "w") as output:
    writer = csv.writer(output, delimiter=',', lineterminator='\n')
    for i in range(942):
        writer.writerow(i)
        writer.writerow(X_df.iloc[i])

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