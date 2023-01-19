#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

while True:
    line = sys.stdin.readline()
    if not line:
        break
    fields = line.rstrip().split(',')
    if fields[-1] == '0':
        fields[-1] = 'not sexist'
    elif fields[-1] == '1':
        fields[-1] = 'sexist'
    print(','.join(fields))
