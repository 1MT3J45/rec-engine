import pandas as pd
import pickle, csv
# --------------------------
from surprise import Reader, Dataset, evaluate, print_perf, KNNBasic
from collections import defaultdict
import numpy as np

fp = open("shared.pkl")
shared = pickle.load(fp)
print "VALUE: ",shared[2:4], "\n TYPE: ", type(shared)

uid = [shared[0], shared[2]]
cid = [shared[1], shared[3]]

# TODO - Import All Clusters

cl0 = pd.read_csv("cluster0.csv", sep=',')
cl1 = pd.read_csv("cluster1.csv", sep=',')
cl2 = pd.read_csv("cluster2.csv", sep=',')

# TODO - Add Cluster identified data to resp. cluster
cld = pd.read_csv("input.csv", sep=',')

# changing CLD into combined data based on previous results
df_u1 = cld[cld['uid']==uid[0]]
df_u2 = cld[cld['uid']==uid[1]]

flag1, flag2 = 0, 0
print "\n\t\t CLUSTER REC" \
      "\n\tMODE: \n\t" \
      "\n\t(1) User Based" \
      "\n\t(2) Item Based"
mode = input("Mode no.:")
if mode is 1:
     mode = True
elif mode is 2:
     mode = False

print "\n\tALGO: \n\t" \
       "\n\t(1) Vector Cosine" \
       "\n\t(2) Pearson Co-relation"
algo = input("Algo no.:")
if algo is 1:
    algo = "cosine"
elif algo is 2:
    algo = "pearson"

# for df_u1 i.e First User in List
# ------------------------------------------------------------------- PRE PROCESSING
if cid[0] == 0:
    pre_df01 = cl0.append(df_u1, ignore_index=True); print "U1 in CL0"
    flag1 = 1
elif cid[0] == 1:
    pre_df02 = cl1.append(df_u1, ignore_index=True); print "U1 in CL1"
    flag1 = 2
elif cid[0] == 2:
    pre_df03 = cl2.append(df_u1, ignore_index=True); print "U1 in CL2"
    flag1 = 3
else:
    print "Exception occurred"
    exit(0)

# for df_u2 i.e Second User in List
if cid[1] == 0:
    pre_df11 = cl0.append(df_u2, ignore_index=True); print "U2 in CL0"
    flag2 = 1
elif cid[1] == 1:
    pre_df12 = cl1.append(df_u2, ignore_index=True); print "U2 in CL1"
    flag2 = 2
elif cid[1] == 2:
    pre_df13 = cl2.append(df_u2, ignore_index=True); print "U2 in CL2"
    flag2 = 3
else:
    print "Exception occurred"
    exit(0)

# Switching on only pre_df's that are created
# Fetching only created DFs
post_df1 = pd.DataFrame
post_df2 = pd.DataFrame
# --------------------------------------------------------------------- POST PROCESSING
# PRE PROCESSED CLUSTER 0 -- Named to POST DataFrame1
if flag1 is 1:
    print pre_df01
    post_df1 = pre_df01 #.iloc[0:, :]
    print "post_df01 ---- PRINTED"
elif flag1 is 2:
    print pre_df02
    post_df1 = pre_df02 #.iloc[0:, :]
    print "post_df02 ---- PRINTED"
elif flag1 is 3:
    print pre_df03
    post_df1 = pre_df03 #.iloc[0:, :]
    print "post_df03 ---- PRINTED"

# ----------------------------------------------------- EXPORTING POST PRO C0
post_df1.to_csv("C0-Processed.csv",sep=',', index_label=False, index=False)

# PRE PROCESSED CLUSTER 1 -- Named to POST DataFrame2
if flag2 is 1:
    print pre_df11
    post_df2 = pre_df11
    print "post_df11 ---- PRINTED"
elif flag2 is 2:
    print pre_df12
    post_df2 = pre_df12
    print "post_df12 ---- PRINTED"
elif flag2 is 3:
    print pre_df13
    post_df2 = pre_df13
    print "post_df13 ---- PRINTED"

# ----------------------------------------------------- EXPORTING POST PRO C1
post_df2.to_csv("C1-Processed.csv", sep=',', index_label=False, index=False)

tup1 = post_df1.shape
tup2 = post_df2.shape
size_of1 = tup1[0]
size_of2 = tup2[0]
# APPENDING UNIT TIMESTAMP TO EXPORT USING SurPRISE PKG
post_df1 = np.append(arr=post_df1, values=np.ones((size_of1,1)).astype(int), axis=1)
post_df2 = np.append(arr=post_df2, values=np.ones((size_of2,1)).astype(int), axis=1)
# EXPORT TO CSV & LOAD AGAIN IN PROGRAM
np.savetxt('po_cluster0.csv', post_df1, delimiter='\t')
np.savetxt('po_cluster1.csv', post_df2, delimiter='\t')


# Predicting Missing Data / NaN Values
# ------------------------------------------------------------- TRAIL & ERROR

def get_top_n(predictions, n=5):            # ======== FUNCTION START
    top_n = defaultdict(list)
    uid = None

    # MAPPING PREDICTIONS TO EACH USER
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # THEN SORT THE PREDICTIONS FOR EACH USER AND RETRIEVE THE K Highest ones
    # uid = 0
    for iid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n                            # ======== FUNCTION END

sim_op = {'name': algo, 'user_based': mode}
algo = KNNBasic(sim_options=sim_op)

reader = Reader(line_format="user item rating", sep='\t')
df0 = Dataset.load_from_file('po_cluster0.csv', reader=reader)
df1 = Dataset.load_from_file('po_cluster1.csv', reader=reader)
# START TRAINING
trainset = df0.build_full_trainset()

# APPLYING ALGORITHM KNN Basic
algo.train(trainset)
print "ALGORITHM USED: \n", algo

testset = trainset.build_anti_testset()
predictions = algo.test(testset=testset)

top_n = get_top_n(predictions, 5)

# # ---------------------------------------------------- PREDICTION VERIFICATION - CL0 (945)
# print "\t\tINITIATING IN CLUSTER 0 (945)\n"
# search_key = raw_input("Enter User ID:")
# item_id = raw_input("Enter Item ID:")
# actual_rating = input("Enter actual Rating:")
#
# print algo.predict(str(search_key), item_id, actual_rating)
#
# # ---------------------------------------------------- PREDICTION VERIFICATION - CL1 (944)
# print "\t\tINITIATING IN CLUSTER 1 (944)\n"
# search_key = raw_input("Enter User ID:")
# item_id = raw_input("Enter Item ID:")
# actual_rating = input("Enter actual Rating:")
#
# print algo.predict(str(search_key), item_id, actual_rating)

# --------------------- GENERATE FULL PREDICTION
csvfile = 'pred_matrix-Cluster0.csv'
with open(csvfile, "w") as output:
    writer = csv.writer(output, delimiter=',', lineterminator='\n')
    writer.writerow(['uid', 'iid', 'rat'])
    for uid, user_ratings in top_n.items():
        for (iid, r) in user_ratings:
            value = uid, iid, r
            writer.writerow(value)
print "\nPROCESSING COMPLETE: \nCheck files with Name\n" \
      "C0-Processed.csv\n" \
      "C1-Processed.csv"