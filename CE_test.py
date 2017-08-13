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

# choice = int(input("\n\t\tMERGE FRAMES\n\t\t1) IBCF + U.DATA\n\t\t2) UBCF + U.DATA\n\t\tEnter your Choice:"))
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
AD_Matrix = AllData.drop_duplicates(subset=['uid','iid'])
Pivot_Matrix = AD_Matrix.pivot(values='rat',index='uid',columns='iid')

#if choice is 1:
#    Pivot_Matrix.to_csv("MainMatrix_IBCF.csv")
#elif choice is 2:
#    Pivot_Matrix.to_csv("MainMatrix_UBCF.csv")
# AD_Matrix.reset_index().pivot(values=3, index=[0, 1], columns=2, aggfunc='mean')

# TODO ----------------- STAGE 2 Prep-ing data for Clustering

records = int(input("Enter the no. of Records to be Fetched:"))
df = pd.read_csv("AllData.csv", usecols=['iid','rat'], nrows=records)

print "UDATA -------------\n", udata_df.dtypes
print "PRED_MATRIX -------\n", pred_matrix.dtypes
print "AllDATA -----------\n", AllData.dtypes
# a = raw_input("awaiting to clean records from AllData.csv! Go ahead.")
X = df.values
# Data type must of Same type, not Mixed type TO BE RESOLVED

# TODO ----------------- STAGE 3 KMeans Clustering

# someArray = df[['uid','iid','rat']].values
# df.loc[:, ['iid','rat']].values
# print "PRINTING X Data \n",X
# print "PRINTING Y Data \n",Y

kmeans = KMeans(n_clusters=3).fit(X)
# KMeans.fit(X)

centroids = kmeans.cluster_centers_
label = kmeans.labels_

print "CENTROIDS :", centroids
print "LABEL :",label

colors = np.random.rand(50)# {"g", "r", "b"}

for i in range(len(X)):
    print "\nCoordinate", X[i], "\nlabel:", label[i]
    plt.plot(X[i][0], X[i][1], markersize = 100)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths= 5, zorder = 10)
print "CO-ORDINATES for Centroids:\n", "1st Centroid", centroids[[0]], "\n2nd Centroid", centroids[[1]], \
    "\n3rd Centroid",centroids[[2]]
plt.show()

# TODO -------------- STAGE 4 Predict Cluster for new Data
iid = int(input("Enter Item ID:"))
rat = float(input("Enter Rating:"))
print "Cluster Label",kmeans.predict([iid, rat])

# NULL'''