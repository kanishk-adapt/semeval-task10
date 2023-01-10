#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2021, 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on
# https://github.com/jowagner/CA4023-NLP/blob/main/notebooks/sentiment-naive-bayes.ipynb

from __future__ import print_function

import sys


class SexismDetectorInterface:

    def __init__(self, task = 'a'):
        self.task = task

    def train(self, data_with_labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError



