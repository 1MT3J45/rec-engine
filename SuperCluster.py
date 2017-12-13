import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from SandBox import Predictor

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
sucl0 = Super_Clusters[0]
sucl0.to_csv('scul0.csv', index=False)
sucl0 = pd.read_csv('scul0.csv')
print sucl0.pivot(index='uid', columns='iid', values='rat')
print sucl0


sucl1 = Super_Clusters[1]
sucl1.to_csv('scul1.csv', index=False)
sucl1 = pd.read_csv('scul1.csv')
print sucl1.pivot(index='uid', columns='iid', values='rat').head(2)
print sucl1.head()

sucl2 = Super_Clusters[2]
sucl2.to_csv('scul2.csv', index=False)
sucl2 = pd.read_csv('scul2.csv')
print sucl2.pivot(index='uid', columns='iid', values='rat').head(2)
print sucl2.head()

sucl3 = Super_Clusters[3]
sucl3.to_csv('scul3.csv', index=False)
sucl3 = pd.read_csv('scul3.csv')
print sucl3.pivot(index='uid', columns='iid', values='rat').head(2)
print sucl3.head()

sucl4 = Super_Clusters[4]
sucl4.to_csv('scul4.csv', index=False)
sucl4 = pd.read_csv('scul4.csv')
print sucl4.pivot(index='uid', columns='iid', values='rat').head(2)
print sucl4.head()

cluster0 = np.append(arr=sucl0, values=np.ones((sucl0.__len__(), 1)).astype(int), axis=1)
cluster1 = np.append(arr=sucl1, values=np.ones((sucl1.__len__(), 1)).astype(int), axis=1)
cluster2 = np.append(arr=sucl2, values=np.ones((sucl2.__len__(), 1)).astype(int), axis=1)
cluster3 = np.append(arr=sucl3, values=np.ones((sucl3.__len__(), 1)).astype(int), axis=1)
cluster4 = np.append(arr=sucl4, values=np.ones((sucl4.__len__(), 1)).astype(int), axis=1)


def pre_process_cluster(df):

    p_df = np.append(arr=df, values=np.ones((df.__len__(), 1)).astype(int), axis=1)
    p_df = pd.DataFrame(p_df)
    p_df = p_df.dropna()
    return p_df

algo = int(input("Select Algorithm for Value Prediction \n\t1. Vector Cosine\n\t2. Pearson"))
A= ''
if algo is 1:
    A = 'cosine'
elif algo is 2:
    A = 'pearson'

try:
    choice = int(input("Select A Cluster for Prediction of Missing Values: \n Clusters Available (0 - 4)\n Enter your"
                       " Choice:"))
    if choice is 0:
        cluster = pre_process_cluster(sucl0)
        path = 'scul0.csv'
        p = Predictor()
        p.choices(algorithm=A, processed_df=cluster, df_path=path)
    elif choice is 1:
        cluster = pre_process_cluster(sucl1)
        path = 'scul1.csv'
        p = Predictor()
        p.choices(algorithm=A, processed_df=cluster, df_path=path)
    elif choice is 2:
        cluster = pre_process_cluster(sucl2)
        path = 'scul2.csv'
        p = Predictor()
        p.choices(algorithm=A, processed_df=cluster, df_path=path)
    elif choice is 3:
        cluster = pre_process_cluster(sucl3)
        path = 'scul3.csv'
        p = Predictor()
        p.choices(algorithm=A, processed_df=cluster, df_path=path)
    elif choice is 4:
        cluster = pre_process_cluster(sucl4)
        path = 'scul4.csv'
        p = Predictor()
        p.choices(algorithm=A, processed_df=cluster, df_path=path)
except Exception as e:
    print "CHOICE ERROR:",e
