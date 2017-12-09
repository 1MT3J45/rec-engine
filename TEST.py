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
    print "Caution:",e

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

try:
    input_df = input_df.fillna(0)
    young_Clusters = recipe_one(dataset=input_df)

except Exception as e:
    print "NOTE", e

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

def predictor(clusters):
    # np.append(arr=post_df1, values=np.ones((size_of1,1)).astype(int), axis=1)
    # SEGREGATE CLUSTERS
    cluster0 = clusters[0]
    cluster1 = clusters[1]
    cluster2 = clusters[2]
    cluster3 = clusters[3]
    cluster4 = clusters[4]

    cluster0 = np.append(arr=cluster0, values=np.ones((cluster0.__len__(),1)).astype(int), axis=1)
    cluster1 = np.append(arr=cluster1, values=np.ones((cluster1.__len__(), 1)).astype(int), axis=1)
    cluster2 = np.append(arr=cluster2, values=np.ones((cluster2.__len__(), 1)).astype(int), axis=1)
    cluster3 = np.append(arr=cluster3, values=np.ones((cluster3.__len__(), 1)).astype(int), axis=1)
    cluster4 = np.append(arr=cluster4, values=np.ones((cluster4.__len__(), 1)).astype(int), axis=1)

    choice = int(input("Select A Cluster for Prediction of Missing Values: \n Clusters Available (0 - 5)\n Enter your"
                       " Choice:"))
# yc0.reset_index(inplace=True) # to get UID in dataframe
# fc0.loc[yc0.index, yc0.columns] = yc0 # Match columns of UID from dataframe and add values
#
# input_df0 = input_df[yc0.uid]
# temp = pd.concat([fc0, yc0.index(fc0.columns.__len())]) # Merge in Dataframes in Matrix itself