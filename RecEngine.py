from surprise import Reader, Dataset, evaluate, print_perf, KNNBasic
from collections import defaultdict
import io
import pandas as pd
import csv
from recsys.evaluation.decision import PrecisionRecallF1
import time
import numpy as np

headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

# LOADING DATA-SET
df = pd.read_csv('ml-100k/u.data', sep='\t', names=('UserID', 'MovieID', 'Rating', 'TimeStamp'))
print df.dtypes, '\n', '____________________'


# PRINTING ONLY SPECIFIC COLUMNS
values = df.values
print values[:, 0:2]

# MAKING A MATRIX
df_sorted_values = df.sort_values(['UserID', 'MovieID'])
print type(df_sorted_values)

print("*___________________*")
df_matrix = df.pivot_table(values='Rating', index='UserID', columns='MovieID')
df_matrix.to_csv("Basic_matrix.csv", sep="\t")
# FOLLOWING STATEMENTS PRODUCE THE SAME RESULT

# df3 = df.set_index(['UserID','MovieID'])['Rating'].unstack()
# df3 = df.pivot(index='UserID',columns='MovieID',values='Rating')
# df3 = df.groupby(['UserID','MovieID'])['Rating'].mean().unstack()
print df_matrix


# ----------------------------------------------------------------SPLITTER
def splitter(fold, dataset):
    start = time.time()
    dataset.split(fold)
    end = time.time()
    # dataset.build_full_trainset()
    print "fold =", fold, "Time elapsed =", (end-start)
# -----------------------------------------------------------------SPLITTER ENDS


# ----------------------------------------------------------------------------UBCF
def user_based_cf(co_pe):
    # INITIALIZE REQUIRED PARAMETERS
    # path = 'ml-100k/u.user'
    prnt = "USER"
    sim_op = {'name': co_pe, 'user_based': True}
    algo = KNNBasic(sim_options=sim_op)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    algo.train(trainset)
    print "ALGORITHM USED", co_pe

# -------------------------------`-------------- MARKERS

    f = io.open("_AlgoHist_ub.txt", "wb")
    f.write(repr(co_pe))
    f.close()

# --------------------------------------------- MARKERS END

    print "CF Type:", prnt, "BASED"

    # PEEKING PREDICTED VALUES
    search_key = raw_input("Enter User ID:")
    item_id = raw_input("Enter Item ID:")
    actual_rating = input("Enter actual Rating:")

    print algo.predict(str(search_key), item_id, actual_rating)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset=testset)

    top_n = get_top_n(predictions, 5)
    result_u = True

    k = input("Enter size of Neighborhood (Min:1, Max:40)")

    inner_id = algo.trainset.to_inner_iid(search_key)
    neighbors = algo.get_neighbors(inner_id, k=k)
    print "Nearest Matching users are:"
    for i in neighbors:
        print "\t "*6, i
    return top_n, result_u
# --------------------------------------------------------- RECOMMENDING MOVIES TO SPECIFIC USER


# ----------------------------------------- GENERATION OF PREDICTION MATRIX
def gen_pred_matrix_ubcf(co_pe):

    # ---------------------------------------------------- UBCF as is

    # INITIALIZE REQUIRED PARAMETERS
    # path = 'ml-100k/u.user'
    prnt = "USER"
    sim_op = {'name': co_pe, 'user_based': True}
    algo = KNNBasic(sim_options=sim_op)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    algo.train(trainset)
    print "ALGORITHM USED", co_pe

    print "CF Type:", prnt, "BASED"

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset=testset)

    top_n = get_top_n(predictions, 5)

    # ---------------------------------------------------- UBCF as is

    csvfile = 'pred_matrix-full_ubcf.csv'
    with open(csvfile, "w") as output:
        writer = csv.writer(output, delimiter=',', lineterminator='\n')
        writer.writerow(['uid', 'iid', 'rat'])
        for uid, user_ratings in top_n.items():
            for (iid, r) in user_ratings:
                value = uid, iid, r
                writer.writerow(value)
    print "Done! You may now check the file in same Dir. as of Program"


def gen_pred_matrix_ibcf(co_pe):
    # ---------------------------------------------------- IBCF as is

    # INITIALIZE REQUIRED PARAMETERS
    path = '/home/mister-t/Projects/PycharmProjects/RecommendationSys/ml-100k/u.item'
    prnt = "ITEM"
    sim_op = {'name': co_pe, 'user_based': False}
    algo = KNNBasic(sim_options=sim_op)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    res = algo.train(trainset)
    print "\t\t >>>TRAINED SET<<<<\n\n", res

    # Read the mappings raw id <-> movie name
    # rid_to_name, name_to_rid = read_item_names(path)
    print "CF Type:", prnt, "BASED"
    print "Please be Patient while 'pred_matrix-full_ibcf.csv' is being Generated"
    for i in range(5):
        print "."
        time.sleep(0.5)
    # --------------------------------------------------------- EXPERIMENTAL

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset=testset)

    top_n = get_top_n(predictions, 5)

    # --------------------------------------------------------- EXPERIMENTAL

    # ---------------------------------------------------- IBCF as is

    csvfile = 'pred_matrix-full_ibcf.csv'
    with open(csvfile, "w") as output:
        writer = csv.writer(output, delimiter=',', lineterminator='\n')
        writer.writerow(['uid', 'iid', 'rat'])
        for uid, user_ratings in top_n.items():
            for (iid, r) in user_ratings:
                value = uid, iid, r
                writer.writerow(value)
    print "Done! You may now check the file in same Dir. as of Program"

# ----------------------------------------- GENERATION OF PREDICTION MATRIX ENDS


def get_top_n(predictions, n=5):
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
    return top_n

    # --------------------------------------------------------- END RECOMMENDATION


# ------------------------------------------------------------------UBCF ENDS


# ------------------------------------------------------------------READ ITEM NAMES
def read_item_names(path):
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids."""

    file_name = path
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid
# -------------------------------------------------------------------READ ITEM NAMES ENDS


# ----------------------------------------------------------------------------IBCF
def item_based_cf(co_pe):
    # INITIALIZE REQUIRED PARAMETERS
    # INITIALIZE REQUIRED PARAMETERS
    path = '/home/mister-t/Projects/PycharmProjects/RecommendationSys/ml-100k/u.item'
    prnt = "ITEM"
    sim_op = {'name': co_pe, 'user_based': False}
    algo = KNNBasic(sim_options=sim_op)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    res = algo.train(trainset)
    print "\t\t >>>TRAINED SET<<<<\n\n", res

    # Read the mappings raw id <-> movie name
    rid_to_name, name_to_rid = read_item_names(path)
    print "CF Type:", prnt, "BASED"

    search_key = raw_input("Movie Name:")
    print "ALGORITHM USED : ", co_pe
    raw_id = name_to_rid[search_key]

    # --------------------------------------------- MARKERS

    f = io.open("_AlgoHist_ib.txt", "wb")
    f.write(repr(co_pe))
    f.close()

    # --------------------------------------------- MARKERS END

    print "\t\t RAW ID>>>>>>>", raw_id, "<<<<<<<"
    inner_id = algo.trainset.to_inner_iid(raw_id)

    print "INNER ID >>>>>", inner_id

    # Retrieve inner ids of the nearest neighbors of Toy Story.
    k = input("Enter size of Neighborhood (Min:1, Max:40)")
    neighbors = algo.get_neighbors(inner_id, k=k)

    neighbors = (algo.trainset.to_raw_iid(inner_id)
                 for inner_id in neighbors)
    neighbors = (rid_to_name[rid]
                 for rid in neighbors)

    print "Nearest ", k, " Matching Items are:"
    for i in neighbors:
        print "\t " * 6, i


# ---------------------------------------------------------------------IBCF ENDS
# METHODS DEFINED PRIOR


# ---------------------------------------------------------------------UBCF EVAL TEST
def ubcf_eval(co_pe):
    kfold = input("Enter number of folds required to Evaluate:")

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    splitter(kfold, df)

    # SIMILARITY & ALGORITHM DEFINING
    sim_op = {'name': co_pe, 'user_based': True}
    algo = KNNBasic(sim_options=sim_op)

    # RESPONSIBLE TO EXECUTE DATA SPLITS MENTIONED IN STEP 4
    start = time.time()
    perf = evaluate(algo, df, measures=['RMSE', 'MAE'])
    end = time.time()

    print_perf(perf)

    print "\nTotal Time elapsed =", (end - start)
    print "Average time per fold =", (end - start)/kfold, "\n"

    print perf

    ds = pd.read_csv("pred_matrix-full_ubcf.csv")
    confusion_matrix = np.matrix(ds)

    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    print "\nTrue Positive:\n", TP, "\n\nTrue Negative\n", TN, "\n\nFalse Positive\n", FP, "\n\nFalse Negative\n", FN
    print "-" * 30
    print "\nTrue Postive Ratio =", TPR, "\n\nFalse Positive Ratio =", FPR
    print "-" * 30

    print "*" * 20
    print confusion_matrix

    print "Accuracy with current Algorithm", algo, "is ", ACC.mean(axis=0)


# ---------------------------------------------------------------------UBCF EVAL TEST ENDS


# ---------------------------------------------------------------------IBCF EVAL TEST
def ibcf_eval(co_pe):
    kfold = input("Enter number of folds required to Evaluate:")

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    splitter(kfold, df)

    # SIMILARITY & ALGORITHM DEFINING
    sim_op = {'name': co_pe, 'user_based': False}
    algo = KNNBasic(sim_options=sim_op)

    # RESPONSIBLE TO EXECUTE DATA SPLITS MENTIONED IN STEP 4
    start = time.time()
    perf = evaluate(algo, df, measures=['RMSE', 'MAE'],)
    end = time.time()

    print_perf(perf)

    print "\nTotal Time elapsed =", (end - start)
    print "Average time per fold =", (end - start)/kfold, "\n"

    return perf
# ---------------------------------------------------------------------IBCF EVAL TEST ENDS


# --------------------------------------------------------------------- RECOMMEND MOVIES
def recommend(query, algo):

    print "\n\t\t\t Please select the MODE of Recommendation \n\t\t\t(1) UBCF Recommendations " \
          "\n\t\t\t(2) IBCF Recommendations \n\t\t\t (0) To exit to Main Menu"

    df, hist = None, None

    ch = int(input("Ch >"))
    if ch is 1:
        df = pd.read_csv("pred_matrix-full_ubcf.csv")
        hist = io.open("AlgoHist_ub.txt", "r")
    elif ch is 2:
        df = pd.read_csv("pred_matrix-full_ibcf.csv")
        hist = io.open("AlgoHist_ib.txt", "r")
    elif ch is 0:
        df = None
        hist = None
        choices(algo)

    query = int(query)
    samsize = int(input("Enter the sample size:"))
    print ("NOTE: This process will take time to execute. Pls. be Patient \n")

    start = time.time()

    records = df[df['uid'] == query].head(samsize)
    df_movies = pd.read_csv('ml-100k/u.item', sep='|')  # DF3 Items| ID | MOVIE NAME (YEAR)| REL.DATE | NULL | IMDB LINK|
    records['iid'] = records['iid'].replace(df_movies.set_index('ID')['MOVIE NAME (YEAR)'])

    print records
    end = time.time()
    print "-"*15, "SUMMARY", "-"*15
    print "Total Time taken\t\t\t", (end - start)/60
    print "Total Records traversed\t\t", df.shape
    print "Algorithm used:\t\t\t\t", hist.read()
    hist.close()

# --------------------------------------------------------------------- RECOMMEND MOVIES ENDS


def pre_rec_graph(dataset):
    print "-" * 15, "GRAPH GEN W/ PRECISION RECALL", "-"*15
    df1 = pd.read_csv("pred_matrix-full_ubcf.csv", sep=",")
    df2 = pd.read_csv("pred_matrix-full_ibcf.csv", sep=",")

    ub_pred = list(df1.ix[::, 'rat'])
    ib_pred = list(df2.ix[::, 'rat'])
    ds_prev = list(dataset.ix[::, 'Rating'])

    decision = PrecisionRecallF1()
    decision.load(ground_truth=ub_pred, test=ib_pred)
    result = decision.compute()
    print "PRF1 w.r.t. IBCF to UBCF Pred Matrix\nRECALL:", result[0], "\nPRECISION:", result[1], "\nF1:", result[2], "\n"

    decision.load(ground_truth=ds_prev, test=ib_pred)
    result = decision.compute()
    print "PRF1 w.r.t. IBCF Pred Matrix to U.DATA \nRECALL:", result[0], "\nPRECISION:", result[1], "\nF1:", result[2]
    print "\n"


def choices(algorithm):
    print "CHOOSE Relevant option no.\n(1) Predict Rating for User or Movie \n(2) Evaluate performance of Prediction" \
          " \n(3) Generate Prediction Matrix\n(4) Get Recommendation\n(5) Precision & Recall \n(6) Press 6 to exit\n"
    choice = int(input("Choice:"))

    menu_length = 6  # Increment by the number of options you add in Menu (inclusive of Exit)

    if choice == 0:  # ------------------------------------------------- Only Integers to be accepted
        print "Try appropriate options!"
    else:
        while choice <= menu_length:  # ---------------------------------------------- LOOPING CHOICE (PREDICTION MENU)
            if choice == 1:  # --------------------------------------------------- (1) PREDICT RATING FOR USER OR MOVIE
                print "\n\t\tPrediction Menu:\n\t\t1. User Based\n\t\t2. Item Based\n\t\tType 0 to exit\n\t\t"
                print "\t\tTRIGGERS:\n\t\t Algorithm:", algorithm
                ch1 = input("Choice:")

                if ch1 == 1:
                    user_based_cf(algorithm)
                elif ch1 == 2:
                    item_based_cf(algorithm)
                elif ch1 == 0:
                    choices(algorithm)
                else:
                    print "Try Choosing appropriate number.\n exiting..."
                    exit(0)

# ----------------------------------------------------------------------------------- LOOPING CHOICE (EVALUATION MENU)
            elif choice == 2:  # ------------------------------------------------------(2) EVALUATE PERF. OF PREDICTION:
                print "\n\t\tEvaluation Menu:\n\t\t1. User Based Eval\n\t\t2. Item Based Eval\n\t\tType 0 to exit\n\t\t"
                print "TRIGGERS:\n Algorithm:", algorithm
                ch2 = input("Choice:")

                if ch2 == 1:
                    ubcf_eval(algorithm)
                elif ch2 == 2:
                    ibcf_eval(algorithm)
                elif ch2 == 0:
                    choices(algorithm)
                else:
                    print "Try Choosing appropriate number.\n exiting..."
                    exit(0)

            elif choice == 3:
                print "\n\t\t Matrix Generators:\n\t\t 1. UBCF Pred. Matrix\n\t\t 2. IBCF Pred. Matrix\n\t\t 3. " \
                      "Type 0 to Exit\n\t\tNote: Execution of UBCF & IBCF execution is mandatory before generating " \
                      "Resp. Prediction Matrices"
                ch3 = input("Ch >")

                if ch3 == 1:
                    gen_pred_matrix_ubcf(algorithm)
                elif ch3 == 2:
                    gen_pred_matrix_ibcf(algorithm)
                else:
                    choices(algorithm)

            elif choice == 4:
                print "\n\t\t Recommend top n Movies to a Specific user"
                # TWO OPTIONS TO BE MODIFIED INTO 1> Get Recommendations 2> Evaluate Recommendations
                print "\n\t\tReady? \n\t\t (1) to start \n\t\t (2) to exit\n\t\t"
                ch4 = input("Ch >")
                if ch4 == 1:
                    uid = raw_input("Enter User ID:")
                    recommend(uid, algorithm)
                elif ch4 == 2:
                    choices(algorithm)
                else:
                    print "SELECT APPROPRIATE OPTIONS!"
                    choices(algorithm)
            elif choice == 5:
                pre_rec_graph(df)
                choices(algorithm)
            elif choice == 6:
                print "Exiting..."
                exit(0)
        else:
            pass


# -------------- RECOMMENDATION & EVALUATION with PRECISION & RECALL-----


# ---------------------- INITIALIZATION

# INITIAL ALGORITHMS SELECTED WILL BE RIGID ONCE SELECTED:
ip = raw_input("Enter the choice of algorithm(1. Cosine/2. Pearson):")
one = None
if ip == '1':
    one = 'cosine'
elif ip == '2':
    one = 'pearson'
else:
    one = 'None'
    print "Please retry with proper option!"
    exit(0)

# ---------------------- EXPERIMENTAL !

try:
    choices(one)

except:
    print "Exit Exception: Bye"