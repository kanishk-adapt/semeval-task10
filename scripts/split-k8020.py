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

k80len = int(round(len(df) * 0.8))

k8020_tr = df.loc[:k80len-1]   # https://pandas.pydata.org/docs/user_guide/advanced.html#advanced-endpoints-are-inclusive
k8020_dev = df.loc[k80len:]

print('tr size:', len(k8020_tr), 'expected:', k80len)

# https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file
k8020_tr.to_csv( 'k8020-tr.csv',  encoding='utf-8', index=False)
k8020_dev.to_csv('k8020-dev.csv', encoding='utf-8', index=False)

