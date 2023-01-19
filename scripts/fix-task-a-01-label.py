#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

# fix header
header = sys.stdin.readline().rstrip().split(',')
assert header[0] == 'rewire_id'
assert 'label' in header[-1].lower()
print('rewire_id,label_pred')

# fix body
while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.rstrip().split(',')
    if fields[-1] == '0':
        label = 'not sexist'
    elif fields[-1] == '1':
        label = 'sexist'
    else:
        label = fields[-1]
    print('%s,%s' %(fields[0], label))
