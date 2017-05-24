import pandas as pd
import sys
import csv

# Reading users file:
# u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
# users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

try:
    reader = pd.read_csv('ml-100k/u.user',delimiter='|',usecols=[0,1,2,4],nrows=3)
    print(type(reader),"\n\n")

    print(reader)
    print (range(5))
    # col = int(input("Input number column to print"))
    row = int(input("Input number of rows to print"))
    list1=[0,0,0,0]

    for i in range(3):
        list1[i] = input("Insert val")
        print (i)
        i += 1


    print (pd.read_csv('ml-100k/u.user',delimiter='|',usecols=list1,nrows=row))
    match_list = pd.read_csv(pd.read_csv('ml-100k/u.user',delimiter='|',usecols=list1,nrows=row))
    print ("\n THIS IS NEW MATCH LIST",match_list)
finally:
    exit(0)