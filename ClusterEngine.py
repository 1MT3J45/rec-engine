import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

style.use("ggplot")
print "NOTE:\tTo Generate all CSVs, please run RecEngine First."
print "\t\tFind the RecEngine.py in same directory."
# TODO ----------------- STAGE 1 Merger of Predicted and Rated Values (C)
# ------------------------------------------ Conversion of Data Frames into files
columns = ['uid', 'iid', 'rat']

choice = int(input("\n\t\tMERGE FRAMES\n\t\t1) IBCF + U.DATA\n\t\t2) UBCF + U.DATA\n\t\tEnter your Choice:"))
if choice is 1:
    pred_matrix = pd.read_csv("pred_matrix-full_ibcf.csv", sep=',')
elif choice is 2:
    pred_matrix = pd.read_csv("pred_matrix-full_ubcf.csv", sep=',')
else:
    pred_matrix = None
    print "Select appropriate option! Try again"
    exit(0)

part2 = pd.read_csv("ml-100k/u.data", sep='\t',names=columns)
part2.to_csv("udata.csv", index_label=False)

udata_df = pd.read_csv("udata.csv", skipinitialspace=True, usecols=columns)
print "Exporting CSV, Please be Patient."

# ------------------------------------------ Merging into Main File

with open('AllData.csv', 'w') as handle:
    udata_df.to_csv(handle, index=False)
    pred_matrix.to_csv(handle, index=False)

handle.close()

AllData = pd.read_csv("AllData.csv", low_memory=False)
print AllData.shape
# UBCF SHAPE (9506837, 3)
# IBCF SHAPE (1584472, 3)

# ----------------------------------------- BUILD MAIN MATRIX
# AD_Matrix = AllData.pivot(values='rat',index='uid',columns='iid')
AD_Matrix = AllData.drop_duplicates(subset=['uid','iid'])
Pivot_Matrix = AD_Matrix.pivot(values='rat',index='uid',columns='iid')

if choice is 1:
    Pivot_Matrix.to_csv("MainMatrix_IBCF.csv")
elif choice is 2:
    Pivot_Matrix.to_csv("MainMatrix_UBCF.csv")
# AD_Matrix.reset_index().pivot(values=3, index=[0, 1], columns=2, aggfunc='mean')

# TODO ----------------- STAGE 2 Prep-ing data for Clustering

df = pd.read_csv("AllData.csv")

print "DF ----------------\n", df.dtypes
print "UDATA -------------\n", udata_df.dtypes
print "PRED_MATRIX -------\n", pred_matrix.dtypes
print "AllDATA -----------\n", AllData.dtypes
# Data type must of Same type, not Mixed type TO BE RESOLVED

# TODO ----------------- STAGE 3 KMeans Clustering
'''
values = df.values
print values[:, 1:3]
X = values[:, 1:3]
X_values = np.array(X)

print X_values

kmeans = KMeans(n_clusters=3)
KMeans.fit(X_values)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print centroids
print (labels)

colors = {"g.", "r.", "b."}

for i in range(len(X)):
    print "Coordinate", X[i], "label:", labels[i]
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker= 'x', s=150, linewidths= 5, zorder = 10)
plt.show()'''

# TODO -------------- STAGE 4 Export Clustering
# NULL