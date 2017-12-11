import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:
    ch = int(input("Select Appropriate Matrix\n 1. MainMatrix_IBCF.csv\n 2. MainMatrix_UBCF\n Choice:"))

    if ch is 1:
        data_set = pd.read_csv('MainMatrix_IBCF.csv')
        data_set = data_set.dropna()
        data_set = data_set.set_index('uid')
    elif ch is 2:
        data_set = pd.read_csv('MainMatrix_UBCF.csv')
        data_set = data_set.dropna()
        data_set = data_set.set_index('uid')
    else:
        data_set = pd.DataFrame
except Exception as e:
    print "Caution:", e


def recipe_one(dataset):
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

fc0 = Fresh_Clusters[0].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
fc1 = Fresh_Clusters[1].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
fc2 = Fresh_Clusters[2].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
fc3 = Fresh_Clusters[3].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
fc4 = Fresh_Clusters[4].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')

input_read = pd.read_csv('input.csv')
input_df = input_read.pivot(index='uid', columns='iid', values='rat')

Super_Clusters = []

try:
    input_df = input_df.fillna(0)
    young_Clusters = recipe_one(dataset=input_df)

    yc0 = young_Clusters[0].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
    yc1 = young_Clusters[1].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
    yc2 = young_Clusters[2].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
    yc3 = young_Clusters[3].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
    yc4 = young_Clusters[4].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')

    CL0 = fc0.append(yc0)
    CL1 = fc1.append(yc1)
    CL2 = fc2.append(yc2)
    CL3 = fc3.append(yc3)
    CL4 = fc4.append(yc4)

    Super_Clusters = [CL0, CL1, CL2, CL3, CL4]
except Exception as e:
    print "NOTE", e


for i in range(len(Super_Clusters)):
    Super_Clusters[i] = Super_Clusters[i].replace(0, np.nan)

# TODO -- check NaN in Clusters, execute these lines in Console
scul0 = Super_Clusters[0]
scul0.to_csv('scul0.csv', index=False)
scul0 = pd.read_csv('scul0.csv')
scul0 = scul0.pivot(index='uid', columns='iid', values='rat')
print scul0


scul1 = Super_Clusters[1]
scul1.to_csv('scul1.csv', index=False)
scul1 = pd.read_csv('scul1.csv')
scul1 = scul1.pivot(index='uid', columns='iid', values='rat')
print scul1

scul2 = Super_Clusters[2]
scul2.to_csv('scul2.csv', index=False)
scul2 = pd.read_csv('scul2.csv')
scul2 = scul2.pivot(index='uid', columns='iid', values='rat')
print scul2

scul3 = Super_Clusters[3]
scul3.to_csv('scul3.csv', index=False)
scul3 = pd.read_csv('scul3.csv')
scul3 = scul3.pivot(index='uid', columns='iid', values='rat')
print scul3

scul4 = Super_Clusters[4]
scul4.to_csv('scul4.csv', index=False)
scul4 = pd.read_csv('scul4.csv')
scul4 = scul4.pivot(index='uid', columns='iid', values='rat')
print scul4



def predictor(clusters):
    # np.append(arr=post_df1, values=np.ones((size_of1,1)).astype(int), axis=1)
    # SEGREGATE CLUSTERS
    cluster0 = clusters[0]
    cluster1 = clusters[1]
    cluster2 = clusters[2]
    cluster3 = clusters[3]
    cluster4 = clusters[4]

    cluster0 = np.append(arr=cluster0, values=np.ones((cluster0.__len__(), 1)).astype(int), axis=1)
    cluster1 = np.append(arr=cluster1, values=np.ones((cluster1.__len__(), 1)).astype(int), axis=1)
    cluster2 = np.append(arr=cluster2, values=np.ones((cluster2.__len__(), 1)).astype(int), axis=1)
    cluster3 = np.append(arr=cluster3, values=np.ones((cluster3.__len__(), 1)).astype(int), axis=1)
    cluster4 = np.append(arr=cluster4, values=np.ones((cluster4.__len__(), 1)).astype(int), axis=1)

    try:
        choice = int(input("Select A Cluster for Prediction of Missing Values: \n Clusters Available (0 - 4)\n Enter your"
                           " Choice:"))
        if choice is 0:
            pass
        elif choice is 1:
            pass
        elif choice is 2:
            pass
        elif choice is 3:
            pass
        elif choice is 4:
            pass
    except Exception as e:
        print "CHOICE ERROR:"

    return 0

# -- TODO -- Visualisation, if needed
# wcss = []
# for i in range(1, 7):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 7), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()
#
# plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
# plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
# plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
# plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
# plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Clusters of Users')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.legend()
# plt.show()#
