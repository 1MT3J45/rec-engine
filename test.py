import pandas as pd
import sys
import csv

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
#users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

f = open('ml-100k/u.user', 'r')
try:
    reader = pd.read_csv('ml-100k/u.user',delimiter=',')
    print(u_cols)
    for row in reader:
        print(row)
finally:
    f.close()