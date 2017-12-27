import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from SandBox import Predictor
import matplotlib.pyplot as plt


def pre_process_cluster(df):

    p_df = pd.DataFrame(df)
    p_df = p_df.dropna()
    return p_df
# ---------------------------------------- PRE PROCESS FUNCTION


def recipe_one(dataset, urange):

    wcss = []
    for i in range(1, urange):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X=dataset)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, urange), wcss)
    plt.title("ELBOW CURVE")
    plt.xlabel("No. of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    try:
        no_of_clusters = int(input("Enter Number of Clusters:"))
    except Exception as e:
        print "Invalid Cluster Nos.:", e

    kmeans = KMeans(n_clusters=no_of_clusters)
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
# ---------------------------------------- RECIPE ONE FUNCTION


def check_clusters():

    di = {}
    di.fromkeys(range(Super_Clusters.__len__()))

    for i in range(Super_Clusters.__len__()):
        di[i] = not Super_Clusters[i].empty

    return di
# ---------------------------------------- CHECK CLUSTERS FUNCTION


def fly_to(all_clusters):

    x = pack_clusters()
    cl0, cl1, cl2, cl3, cl4 = None, None, None, None, None
    # truth = check_clusters()
    if x[0].empty is False:
        cl0 = all_clusters[0]
    elif x[1].empty is False:
        cl1 = all_clusters[1]
    elif x[2].empty is False:
        cl2 = all_clusters[2]
    elif x[3].empty is False:
        cl3 = all_clusters[3]
    elif x[4].empty is False:
        cl4 = all_clusters[4]

    try:
        choice = int(input("Select A Cluster for Prediction of Missing Values: \n Clusters Available (0 - 4)\n"
                           " Enter your Choice:"))
        while choice <= 5:
            if choice is 0:
                cluster = pre_process_cluster(cl0)
                path = 'sucl0.csv'
                p = Predictor()
                p.choices(algorithm=A, processed_df=cluster, df_path=path)
            elif choice is 1:
                cluster = pre_process_cluster(cl1)
                path = 'sucl1.csv'
                p = Predictor()
                p.choices(algorithm=A, processed_df=cluster, df_path=path)
            elif choice is 2:
                cluster = pre_process_cluster(cl2)
                path = 'sucl2.csv'
                p = Predictor()
                p.choices(algorithm=A, processed_df=cluster, df_path=path)
            elif choice is 3:
                cluster = pre_process_cluster(cl3)
                path = 'sucl3.csv'
                p = Predictor()
                p.choices(algorithm=A, processed_df=cluster, df_path=path)
            elif choice is 4:
                cluster = pre_process_cluster(cl4)
                path = 'scul4.csv'
                p = Predictor()
                p.choices(algorithm=A, processed_df=cluster, df_path=path)
    except Exception as e:
        print "CHOICE ERROR:", e

# ---------------------------------------- FLY TO FUNCTION


def pack_clusters():

    # Check imports
    try:
        truth = []
        for j in range(5):
            truth.append(not pd.read_csv("sucl%d.csv" % j).empty)
    except IOError:
        pass

    ClusterPacket = []
    if truth[0] is True:
        sucl0 = pd.read_csv('sucl0.csv')
        ClusterPacket.append(sucl0)
    elif truth[1] is True:
        sucl1 = pd.read_csv('sucl1.csv')
        ClusterPacket.append(sucl1)
    elif truth[2] is True:
        sucl2 = pd.read_csv('sucl2.csv')
        ClusterPacket.append(sucl2)
    elif truth[3] is True:
        sucl3 = pd.read_csv('sucl3.csv')
        ClusterPacket.append(sucl3)
    elif truth[4] is True:
        sucl4 = pd.read_csv('sucl4.csv')
        ClusterPacket.append(sucl4)
    else:
        pass

    return ClusterPacket

# ---------------------------------------- PACK CLUSTERS FUNCTION


def kitchen(data_set=pd.DataFrame):
    Fresh_Clusters = recipe_one(dataset=data_set, urange=11)

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
        young_clusters = recipe_one(dataset=input_df, urange=input_df.__len__())

        yc0 = young_clusters[0].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
        yc1 = young_clusters[1].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
        yc2 = young_clusters[2].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
        yc3 = young_clusters[3].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')
        yc4 = young_clusters[4].stack().rename_axis(('uid', 'iid')).reset_index(name='rat')

        cl0 = fc0.append(yc0)
        Super_Clusters.append(cl0)
        cl1 = fc1.append(yc1)
        Super_Clusters.append(cl1)
        cl2 = fc2.append(yc2)
        Super_Clusters.append(cl2)
        cl3 = fc3.append(yc3)
        Super_Clusters.append(cl3)
        cl4 = fc4.append(yc4)
        Super_Clusters.append(cl4)
    except Exception as e:
        print "DIRTY KITCHEN:", e
    return Super_Clusters


def data_menu():
    data_set = 0
    try:
        ch = int(input("Select Appropriate Matrix\n 1. MainMatrix_IBCF\n 2. MainMatrix_UBCF\n 3. Use Existing Clusters"
                       " \nChoice:"))

        if ch is 1:
            self_ds = pd.read_csv('MainMatrix_IBCF.csv')
            self_ds = self_ds.dropna()
            self_ds = self_ds.set_index('uid')
            data_set = self_ds
        elif ch is 2:
            self_ds = pd.read_csv('MainMatrix_UBCF.csv')
            self_ds = self_ds.dropna()
            self_ds = self_ds.set_index('uid')
            data_set = self_ds
        elif ch is 3:
            ClusterPacket = pack_clusters()
            fly_to(ClusterPacket)
        else:
            self_ds = pd.DataFrame
            data_set = self_ds
    except Exception as e:
        print "MENU GARBLED:", e
    return data_set

ds = data_menu()
Super_Clusters = kitchen(data_set=ds)
for i in range(len(Super_Clusters)):
    Super_Clusters[i] = Super_Clusters[i].replace(0, np.nan)

# TODO -- check NaN in Clusters, execute these lines in Console
if not Super_Clusters[0].empty:
    sucl0 = Super_Clusters[0]
    # sucl0 = np.append(arr=sucl0, values=np.ones((sucl0.__len__(), 1)).astype(int), axis=1)
    sucl0 = pd.DataFrame(sucl0).dropna()
    sucl0.to_csv('sucl0.csv', index=False, header=False)
    sucl0 = pd.read_csv('sucl0.csv')
    # print sucl0.pivot(index='uid', columns='iid', values='rat')
    print sucl0

elif not Super_Clusters[1].empty:
    sucl1 = Super_Clusters[1]
    sucl1 = pd.DataFrame(sucl1).dropna()
    sucl1.to_csv('sucl1.csv', index=False, header=False)
    sucl1 = pd.read_csv('sucl1.csv')
    # print sucl1.pivot(index='uid', columns='iid', values='rat').head(2)
    print sucl1.head()

elif not Super_Clusters[2].empty:
    sucl2 = Super_Clusters[2]
    sucl2 = pd.DataFrame(sucl2).dropna()
    sucl2.to_csv('sucl2.csv', index=False, header=False)
    sucl2 = pd.read_csv('sucl2.csv')
    # print sucl2.pivot(index='uid', columns='iid', values='rat').head(2)
    print sucl2.head()

elif not Super_Clusters[3].empty:
    sucl3 = Super_Clusters[3]
    sucl3 = pd.DataFrame(sucl3).dropna()
    sucl3.to_csv('sucl3.csv', index=False, header=False)
    sucl3 = pd.read_csv('sucl3.csv')
    # print sucl3.pivot(index='uid', columns='iid', values='rat').head(2)
    print sucl3.head()

elif not Super_Clusters[4].empty:
    sucl4 = Super_Clusters[4]
    sucl4.to_csv('scul4.csv', index=False, header=False)
    sucl4 = pd.read_csv('scul4.csv')
    # print sucl4.pivot(index='uid', columns='iid', values='rat').head(2)
    print sucl4.head()
else:
    pass

# cluster4 = np.append(arr=sucl4, values=np.ones((sucl4.__len__(), 1)).astype(int), axis=1)


truth = check_clusters()
print "Checking Available Clusters:"
for k, v in truth.iteritems():
    print k, v

algo = int(input("Select Algorithm for Value Prediction \n\t1. Vector Cosine\n\t2. Pearson\n\tch:"))
A = ''
if algo is 1:
    A = 'cosine'
elif algo is 2:
    A = 'pearson'

parcel = pack_clusters()
fly_to(parcel)
