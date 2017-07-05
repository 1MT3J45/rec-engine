from surprise import Reader, Dataset, evaluate, print_perf, KNNBasic
import io
import pandas as pd
import csv
# import graphlab

headers = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

# TODO 1 [DONE] Loading Dataset
df = pd.read_csv('ml-100k/u.data', sep='\t', names=('UserID', 'MovieID', 'Rating', 'TimeStamp'))
print df.dtypes, '\n', '____________________'


# TODO 2 [DONE] Printing only selected columns
values = df.values
print values[:, 0:2]

# TODO 3 [DONE] Simple sorting of values
df_sorted_values = df.sort_values(['UserID', 'MovieID'])
print type(df_sorted_values)

# TODO 4 [DONE] Printing Matrix of Dataset with NaN Values
print("*___________________*")
df_matrix = df.pivot_table(values='Rating', index='UserID', columns='MovieID')

# FOLL STATEMENTS PRODUCE THE SAME RESULT

# df3 = df.set_index(['UserID','MovieID'])['Rating'].unstack()
# df3 = df.pivot(index='UserID',columns='MovieID',values='Rating')
# df3 = df.groupby(['UserID','MovieID'])['Rating'].mean().unstack()
print df_matrix


# ----------------------------------------------------------------SPLITTER
def splitter(fold, dataset):
    dataset.split(fold)
    dataset.build_full_trainset()
    print "fold =",fold
# -----------------------------------------------------------------SPLITTER ENDS
# TODO 6 [DONE] Generate Recommendations based on User/Item for its nearest neighbours


# ----------------------------------------------------------------------------UBCF
def user_based_cf(co_pe):
    # INITIALIZE REQUIRED PARAMETERS
    path = '/home/mister-t/Projects/PycharmProjects/RecommendationSys/ml-100k/u.user'
    prnt = "USER"
    sim_op = {'name': co_pe, 'user_based': True}
    algo = KNNBasic(sim_options=sim_op)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # START TRAINING
    trainset = df.build_full_trainset()

    # APPLYING ALGORITHM KNN Basic
    res = algo.train(trainset)
    print "\t\t >>>TRAINED SET<<<<\n\n", res
    print "ALGORITHM USED", co_pe

    # MARKERS    print "-------------------->3) Choice", choice
    # MARKERS    print "-------------------->Path", path,"\n
    print "CF Type:", prnt, "BASED"

    # PEEKING PREDICTED VALUES
    search_key = raw_input("Enter User ID:")
    item_id = raw_input("Enter Item ID:")
    actual_rating = input("Enter actual Rating:")

    print algo.predict(str(search_key), item_id, actual_rating)
# ------------------------------------------------------------------UBCF ENDS


# ------------------------------------------------------------------READ ITEM NAMES
def read_item_names(path):
    """Read the u.item file from MovieLens 100-k dataset and return two
    mappings to convert raw ids into movie names and movie names into raw ids.
    """

    file_name = (path)
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
# MARKERS    print "-------------------->3) Choice", choice
# MARKERS    print "-------------------->Path", path,"\n
    print "CF Type:", prnt, "BASED"

    search_key = raw_input("ID:")
    print "ALGORITHM USED : ", co_pe
    raw_id = name_to_rid[search_key]
    print "\t\t RAW ID>>>>>>>",raw_id ,"<<<<<<<"
    inner_id = algo.trainset.to_inner_iid(raw_id)
    print "INNER ID >>>>>",inner_id
    # Retrieve inner ids of the nearest neighbors of Toy Story.
    k=5
    neighbors = algo.get_neighbors(inner_id, k=k)

    # Convert inner ids of the neighbors into names.
    neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in neighbors)
    neighbors = (rid_to_name[rid]
                       for rid in neighbors)
    print 'The ', k,' nearest neighbors of ', search_key,' are:'
    for movie in neighbors:
        print(movie)
# ---------------------------------------------------------------------IBCF ENDS
# METHODS DEFINED PRIOR


# ---------------------------------------------------------------------UBCF EVAL TEST
def ubcf_eval(co_pe):
    kfold = input("Enter number of folds required to Evaluate:")

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    splitter(kfold,df)

    # SIMILARITY & ALGORITHM DEFINING
    sim_op = {'name': co_pe, 'user_based': True}
    algo = KNNBasic(sim_options=sim_op)

    # RESPONSIBLE TO EXECUTE DATA SPLITS MENTIONED IN STEP 4
    perf = evaluate(algo, df, measures=['RMSE', 'MAE'], )
    return perf
# ---------------------------------------------------------------------UBCF EVAL TEST ENDS


# ---------------------------------------------------------------------IBCF EVAL TEST
def ibcf_eval(co_pe):
    kfold = input("Enter number of folds required to Evaluate:")
    splitter(kfold)

    reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    # SIMILARITY & ALGORITHM DEFINING
    sim_op = {'name': co_pe, 'user_based': False}
    algo = KNNBasic(sim_options=sim_op)

    # RESPONSIBLE TO EXECUTE DATA SPLITS MENTIONED IN STEP 4
    perf = evaluate(algo, df, measures=['RMSE', 'MAE'], )
    return perf
# ---------------------------------------------------------------------IBCF EVAL TEST ENDS


def choices(algorithm):
    print "CHOOSE Relevant option no.\n(1) Predict Rating for User or Movie \n(2) Evaluate performance of Prediction \n" \
          "(3) Generate Recommendation\n(4) Evaluate Recommendation\n(5) Type 5 to exit \n"
    choice = input("Choice:")

    if type(choice) == type(0): # -------- Only Integers to be accepted
        pass
    else:
        print "Only positive numbers"
        exit(0)

    while choice <= 4:   # ------------------------------------------------------- LOOPING CHOICE (PREDICTION MENU)
        if choice == 1:  # ------------------------------------------------------ (1) PREDICT RATING FOR USER OR MOVIE
            print "\n\t\tPrediction Menu:\n\t\t1. User Based\n\t\t2. Item Based\n\t\tType 0 to exit\n\t\t"
            print "\t\tTRIGGERS:\n\t\t Algorithm:",algorithm
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

        print "choice",choice
    else:
        exit(0)


#       csvfile = 'pred_matrix.csv'
#    with open(csvfile, "w") as output:
#        writer = csv.writer(output,lineterminator='\n')
#        for val in algo.predict(str(range(1,943)),range(1,1683),1):
#            writer.writerow([val])

# # EXECUTION STARTS FROM HERE:
# ---------------------- INITIALIZATION

# TODO 5 [DONE] Similarity using KNN basic after applying Pearson & Cosine
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

    # reader = Reader(line_format="user item rating", sep='\t', rating_scale=(1, 5))
    # df = Dataset.load_from_file('ml-100k/u.data', reader=reader)

    #  CHOICES of Algorithm(one), Dataset(df) are taken before hand
    #  since they will be shared with Evaluation methods as well
    #  Initializing locally to every method would increase the statements


# ---------------------- EXPERIMENTAL

choices(one)