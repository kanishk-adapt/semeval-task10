#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# usage: split-k8020.py train_all_tasks.csv

# warning: writes files k8020-(tr|dev).csv


import pandas
import sys

df = pandas.read_csv(sys.argv[1])

assert len(df) in (14000, 5)

k20len = int(round(len(df) * 0.2))
k40len = int(round(len(df) * 0.4))
k60len = int(round(len(df) * 0.6))
k80len = int(round(len(df) * 0.8))

k8020_tr = df.loc[:k80len-1]   # https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-endpoints-are-inclusive
k8020_dev = df.loc[k80len:]

k8020_dev_run_1 = k8020_dev
k8020_dev_run_2 = df.loc[k60len:k80len-1]
k8020_dev_run_3 = df.loc[k40len:k60len-1]
k8020_dev_run_4 = df.loc[k20len:k40len-1]
k8020_dev_run_5 = df.loc[:k20len-1]


k8020_tr_run_2 = pandas.concat([
    k8020_dev_run_1,
    k8020_dev_run_3,
    k8020_dev_run_4,
    k8020_dev_run_5,
])

k8020_tr_run_3 = pandas.concat([
    k8020_dev_run_1,
    k8020_dev_run_2,
    k8020_dev_run_4,
    k8020_dev_run_5,
])

k8020_tr_run_4 = pandas.concat([
    k8020_dev_run_1,
    k8020_dev_run_2,
    k8020_dev_run_3,
    k8020_dev_run_5,
])

k8020_tr_run_5 = pandas.concat([
    k8020_dev_run_1,
    k8020_dev_run_2,
    k8020_dev_run_3,
    k8020_dev_run_4,
])

print('tr size:', len(k8020_tr), 'expected:', k80len)
print('  run 2:', len(k8020_tr_run_2))
print('  run 3:', len(k8020_tr_run_3))
print('  run 4:', len(k8020_tr_run_4))
print('  run 5:', len(k8020_tr_run_5))

# https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file
k8020_tr.to_csv( 'k8020-tr-run-1.csv',  encoding='utf-8', index=False)
k8020_dev.to_csv('k8020-dev-run-1.csv', encoding='utf-8', index=False)

k8020_tr_run_2.to_csv( 'k8020-tr-run-2.csv',  encoding='utf-8', index=False)
k8020_dev_run_2.to_csv('k8020-dev-run-2.csv', encoding='utf-8', index=False)

k8020_tr_run_3.to_csv( 'k8020-tr-run-3.csv',  encoding='utf-8', index=False)
k8020_dev_run_3.to_csv('k8020-dev-run-3.csv', encoding='utf-8', index=False)

k8020_tr_run_4.to_csv( 'k8020-tr-run-4.csv',  encoding='utf-8', index=False)
k8020_dev_run_4.to_csv('k8020-dev-run-4.csv', encoding='utf-8', index=False)

k8020_tr_run_5.to_csv( 'k8020-tr-run-5.csv',  encoding='utf-8', index=False)
k8020_dev_run_5.to_csv('k8020-dev-run-5.csv', encoding='utf-8', index=False)
