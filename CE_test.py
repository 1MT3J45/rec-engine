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

#choice = int(input("\n\t\tMERGE FRAMES\n\t\t1) IBCF + U.DATA\n\t\t2) UBCF + U.DATA\n\t\tEnter your Choice:"))
choice = 1
print "Choice set to Default = 1 [Calls IBCF]"

if choice is 1:
    pred_matrix = pd.read_csv("pred_matrix-full_ibcf.csv", sep=',')
elif choice is 2:
    pred_matrix = pd.read_csv("pred_matrix-full_ubcf.csv", sep=',')
else:
    pred_matrix = None
    print "Select appropriate option! Try again"
    exit(0)

part2 = pd.read_csv("ml-100k/u.data", sep='\t', names=columns)
part2.to_csv("udata.csv", index_label=False)

udata_df = pd.read_csv("udata.csv", skipinitialspace=True, usecols=columns)
print "Exporting CSV, Please be Patient."

# ------------------------------------------ Merging into Main File

with open('AllData.csv', 'w') as handle:
    udata_df.to_csv(handle, index=False)
    pred_matrix.to_csv(handle, index=False, header=False)
handle.close()

AllData = pd.read_csv("AllData.csv", low_memory=False)
print AllData.shape


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

print "UDATA -------------\n", udata_df.dtypes
print "PRED_MATRIX -------\n", pred_matrix.dtypes
print "AllDATA -----------\n", AllData.dtypes
# Data type must of Same type, not Mixed type TO BE RESOLVED

# TODO ----------------- STAGE 3 KMeans Clustering

np_df = np.genfromtxt('AllData.csv', delimiter=',', unpack=True)

values = df.values

X = values[:, 1:2]
Y = values[:, 2:3]

print df.iloc(128)

'''
X_values = np.array([[242, 3],[302, 3],
                     [377, 1],[51, 2],
                     [346, 1],[474, 4],
                     [265, 2],[465, 5],
                     [451, 3],[86, 3],
                     [257, 2],[1014, 5],
                     [222, 5]])
Y_values = np.array(Y)

someArray = df[['uid','iid','rat']].values
# df.loc[:, ['iid','rat']].values
# print "PRINTING X Data \n",X
# print "PRINTING Y Data \n",Y
print type(X_values)
ch = input("Wait for input ->")
print df
print "Array HERE \n",someArray

kmeans = KMeans(n_clusters=3).fit(X_values)
# KMeans.fit(X)

centroids = kmeans.cluster_centers_
label = kmeans.labels_

print "CENTROIDS :", centroids
print "LABEL :",label

colors = {"g.", "r.", "b."}

for i in range(len(X_values)):
    print "Coordinate", X_values[i], "label:", label[i]
    plt.plot(X_values[i][0], X_values[i][1], markersize = 10)

plt.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=150, linewidths= 5, zorder = 10)
plt.show()

# TODO -------------- STAGE 4 Export Clustering
# NULL'''