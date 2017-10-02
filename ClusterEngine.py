import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import csv, pickle

style.use("ggplot")
print "NOTE:\tTo Generate all CSVs, please run RecEngine First."
print "\t\tFind the RecEngine.py in same directory."
# TODO ----------------- STAGE 1 Merger of Predicted and Rated Values (C)
# ------------------------------------------ Conversion of Data Frames into files
def merge_and_make():
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
    df = pd.read_csv("AllData.csv", usecols=['uid', 'iid', 'rat'])
    print AllData.shape


    # ----------------------------------------- BUILD MAIN MATRIX
    AD_Matrix = AllData.drop_duplicates(subset=['uid', 'iid'])
    Pivot_Matrix = AD_Matrix.pivot(values='rat', index='uid', columns='iid')
    print "Matrix: Generating..."
    if choice is 1:
       Pivot_Matrix.to_csv("MainMatrix_IBCF.csv")
    elif choice is 2:
       Pivot_Matrix.to_csv("MainMatrix_UBCF.csv")
    print "Matrix: Done"

    # TODO ----------------- STAGE 2 Prep-ing data for Clustering

    records = int(input("Enter the no. of Records to be Fetched:"))
    df_x = pd.read_csv("AllData.csv", usecols=['iid', 'rat'], nrows=records)
    print "UDATA -------------\n", udata_df.dtypes
    print "PRED_MATRIX -------\n", pred_matrix.dtypes
    print "AllDATA -----------\n", AllData.dtypes
    # a = raw_input("awaiting to clean records from AllData.csv! Go ahead.")
    X = df_x.values
    print "_______________________________________________________________\n"
    # Data type must of Same type, not Mixed type TO BE RESOLVED

    # TODO ----------------- STAGE 3 KMeans Clustering

    kmeans = KMeans(n_clusters=3).fit(X)

    centroids = kmeans.cluster_centers_
    label = kmeans.labels_

    print "CENTROIDS :", centroids
    print "LABEL :", label

    cl_0, cl_1, cl_2 = [], [], []

    for i in range(len(X)):
        print "\nCoordinate", X[i], "\nlabel:", label[i]
        plt.plot(X[i][0], X[i][1], markersize=100)
        if label[i] == 0:
            print "LABELed 0: ", df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']
            cl_0.append((df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']))
        elif label[i] == 1:
            print "LABELed 1:", df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']
            cl_1.append((df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']))
        else:
            print "LABELed 2:", df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']
            cl_2.append((df.ix[i]['uid'], df.ix[i]['iid'], df.ix[i]['rat']))

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='+', s=150, linewidths= 5, zorder = 10)
    mylist = list(centroids)
    print "CENTROIDS:"
    for i in range(3):
        print "\tX", i+1, "\t\t\tY", i+1
        print mylist[i]
    plt.show()

    print "\n--- CLUSTER 0 Count--- \n", len(cl_0)
    with open('cluster0.csv','w+') as zero:
        writer = csv.writer(zero, delimiter=',', lineterminator='\n')
        writer.writerow(['uid','iid','rat'])
        for _1, _2, _3 in cl_0:
            writer.writerow((_1, _2, _3))
    zero.close()

    print "\n--- CLUSTER 1 Count--- \n", len(cl_1)
    with open('cluster1.csv','w+') as one:
        writer = csv.writer(one, delimiter=',', lineterminator='\n')
        writer.writerow(['uid','iid','rat'])
        for _1, _2, _3 in cl_1:
            writer.writerow((_1, _2, _3))
    one.close()

    print "\n--- CLUSTER 2 Count--- \n", len(cl_2)
    with open('cluster2.csv','w+') as two:
        writer = csv.writer(two, delimiter=',', lineterminator='\n')
        writer.writerow(['uid','iid','rat'])
        for _1, _2, _3 in cl_2:
            writer.writerow((_1, _2, _3))
    two.close()
    print "All Cluster CSVs are now available on Storage disk"
# ------------------------------------------------------------------------- EOM make_cluster()


# TODO -------------- STAGE 4 Predict Cluster for new Data
def predict():

    ip = pd.read_csv("input.csv",sep=",")
    X1 = ip.dropna()
    some_data = []

    kmeans = KMeans(n_clusters=2).fit(X1)

    for i in range(len(X1)):
        some_data = kmeans.predict(X1)

    centroids = kmeans.cluster_centers_
    label = kmeans.labels_

    print "Centroids :\n",centroids
    print "Cluster Nos.:\n", label
    print some_data

    # CREATING TRIGGERS
    z1 = X1['uid'].unique()
    z2 = list(set(some_data))
    pic_file = []

    # TODO -------------- STAGE 6 Sequencing Functions
    for i in range(2):
        print "UID:",z1[i]," is in Cluster", z2[i-1] # z2 is reversed due to sorted data struct. SET is used
        pic_file.append(z1[i])
        pic_file.append(z2[i-1])

    # TODO -------------- STAGE 5 Store the Output to a serialized file
        fp = open("shared.pkl", "w")
        pickle.dump(pic_file, fp)
        print "File PICKLED / SERIALIZED for Cluster Recommendation"

# EXECUTION FLOW
def choice_fun():
    print "\n\t\tSelect appropriate option:\n\t1. Merge & Make Clusters \n\t2. Predict the Cluster of New Users \n\t3. Exit"
    choice = int(input("Ch:"))

    if choice is 1:
        merge_and_make()
        choice_fun()

    elif choice is 2:
        predict()
        choice_fun()

    elif choice is 3:
        print "Exiting..."
        exit(0)

choice_fun()