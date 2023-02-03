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

import numpy
from sklearn import metrics
import sys
import time
from tqdm import tqdm

from edos_labels import task_b_long_labels, task_c_long_labels, task_c_short_labels, label2index


class SexismDetector:

    def __init__(self, task = 'a', model = None):
        self.task = task
        self.model = model
        self.progress_info = True
        self.debug = False

    def train(self, data_with_labels, get_features = False):  # sub-classes may want to add here
        # (1) get descriptive and target features
        tr_features = self.extract_features(
            data_with_labels
        )
        tr_targets = self.get_targets(data_with_labels)
        # (2) train the core model
        self.train_model_on_features(tr_features, tr_targets)
        if get_features:
            return tr_features
        return None

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train self.model
        self.model.fit(tr_features, tr_targets)

    def predict(
        self, data, get_accuracy = False,
        get_confusion_matrix = False,
        use_long_labels = True,
    ):
        features = self.extract_features(data)
        # use numpy to get predictions
        y_pred = self.model.predict(features)
        # restore labels
        labels = []
        for row, label_idx in enumerate(y_pred):
            if self.task == 'a':
                label = 'sexist' if label_idx else 'not sexist'
            elif self.task == 'b' and use_long_labels:
                label = task_b_long_labels[label_idx]
            elif self.task == 'b':
                label = '%d' %(label_idx + 1)
            elif self.task == 'c' and use_long_labels:
                label = task_c_long_labels[label_idx]
            elif self.task == 'c':
                label = task_c_short_labels[label_idx]
            else:
                raise ValueError('unknown task %r' %self.task)
            labels.append(label)
        if get_accuracy or get_confusion_matrix:
            retval = []
            retval.append(labels)
            y_true = self.get_targets(data)
            if get_accuracy:
                retval.append(
                    metrics.accuracy_score(y_true, y_pred)
                )
            if get_confusion_matrix:
                retval.append(
                    metrics.confusion_matrix(y_true, y_pred)
                )
            return tuple(retval)
        else:
            return labels

    def get_vector_length(self):
        ''' return dimension of feature vector for each item '''
        raise NotImplementedError

    def get_vector_dtype(self):  # sub-classes may want to use a different type, e.g. float
        return numpy.int32

    def extract_features(self, data):
        ''' get feature vector for each document in data '''
        # create numpy array of required size
        columns = self.get_vector_length()
        dtype   = self.get_vector_dtype()
        rows = len(data)
        features = numpy.zeros((rows, columns), dtype=dtype)
        # populate feature matrix
        for row, item in enumerate(tqdm(data, desc='Extracting features')):
            self.get_item_feature_vector(item, features, row)
        return features

    def get_item_feature_vector(self, item, matrix = None, row = None):
        raise NotImplementedError

    def get_feature_matrix_column_names(self):
        raise NotImplementedError

    def get_targets(self, data):
        ''' create column vector with target labels
        '''
        global label2index
        # prepare target vector
        targets = numpy.zeros(len(data), dtype=numpy.int8)
        for index, item in enumerate(data):
            label = item.label
            if label == 'none':
                assert self.task == 'a'  # for tasks b and c, "none" items must not be in the data
                # nothing else to do as array is populated with zeroes above
            elif self.task == 'a':
                targets[index] = 1
            elif self.task == 'b':
                # first digit in label indicates coarse sexism category
                coarse_category = int(label[0])
                assert coarse_category >= 1
                targets[index] = coarse_category - 1   # use 0 for first label
            elif self.task == 'c':
                targets[index] = label2index[label]
            else:
                raise ValueError('unsupported task %r' %self.task)
        return targets
