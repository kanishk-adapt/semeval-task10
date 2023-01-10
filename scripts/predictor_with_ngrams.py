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

from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import sys

from predictor_interface import SexismDetectorInterface


fine_grained_labels = '1.1 1.2 2.1 2.2 2.3 3.1 3.2 3.3 3.4 4.1 4.2'.split()

label2index = {}
for index, label in enumerate(fine_grained_labels):
    label2index[label] = index


class SexismDetectorWithVocab(SexismDetectorInterface):

    def __init__(self, tokeniser = None, min_freq = 5, **kwargs)
        super.__init__(self, **kwargs)
        self.tokeniser = tokeniser
        self.min_freq  = min_freq

    def train(self, data_with_labels):
        # (1) build the vocabulary from the training data
        self.reset_vocab()
        self.add_to_vocab_from_data(data_with_labels)
        self.finalise_vocab()
        # (2) get descriptive and target features
        tr_features = self.extract_features(
            data_with_labels
        )
        tr_targets = self.get_targets(data_with_labels)
        # (3) train the core model
        self.train_model_on_features(tr_features, tr_targets)

    # TODO: implement predict() here
        
    def reset_vocab(self):
        self.vocab = defaultdict(lambda: 0)   # for each entry, record number of occurrences
        
    def add_to_vocab_from_data(self, data):
        ''' expand vocabulary to cover new data '''
        for item in data:
            for atom in self.get_item_atoms(item):
                self.vocab[atom] += 1

    def finalise_vocab(self):
        ''' finish creating the vocaulary and create support structures '''
        # apply frequency threshold and reduce to sorted list
        selected_vocab = []
        for entry in self.vocab:
            if self.vocab[entry] >= self.min_freq:
                selected_vocab.append(entry)
        self.vocab = sorted(selected_vocab)
        # create reverse map for fast token lookup
        self.atom2index = {}
        for index, atom in enumerate(self.vocab):
            self.atom2index[atom] = index
    
    def get_vector_length(self):  # sub-classes may want to add components for non-vocab features
        return len(self.vocab)

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
        for row, item in enumerate(data):
            vector = self.get_item_feature_vector(item)
            for column in range(columns):
                features[row, column] = vector[column]
        return features

    def get_item_feature_vector(self, item):  # sub-classes may want to add features here
        columns = self.get_vector_length()
        dtype   = self.get_vector_dtype()
        vector = numpy.zeros((columns,), dtype=dtype)
        for atom in self.get_item_atoms(item):
            try:
                index = self.token2index[atom]
            except KeyError:  # token not in vocab
                continue      # --> skip this atom
            if self.clip_counts:
                vector[index] = 1
            else:
                vector[index] += 1
        return vector

    def get_targets(self, data):
        ''' create column vector with target labels
        '''
        global label2index
        # prepare target vector
        targets = numpy.zeros(len(data), dtype=numpy.int8)
        for index, item in enumerate(data):
            label == item.label
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
    
    # the following functions will have to be implemented in sub-classes
    # to be able to use above functionality

    def get_item_atoms(self, item):
        raise NotImplementedError
    
    def get_targets(self, data):
        raise NotImplementedError

    def train_model_on_features(self, tr_features, tr_targets):
        raise NotImplementedError


class SexismDetectorWithNgrams(SexismDetectorWithVocab):

    def __init__(self, ngram_range = None, padding = None, **kwargs)
        super.__init__(self, **kwargs)
        if not ngram_range:
            ngram_range = [1]
        self.ngram_range = ngram_range
        self.padding = padding

    def get_item_atoms(self, item):
        item_tokens = tokeniser.get_tokens(item.get_text())
        for n in self.ngram_range:
            assert n > 0
            tokens = item_tokens[:]
            if self.padding and n > 1:
                n1_padding = (n-1) * [self.padding]
                tokens = n1_padding + tokens + n1_padding
            start = 0
            while start < seq_length + 1 - n:
                ngram = tokens[start:start+n]
                yield tuple(ngram)
                start += 1


## more material from CA4023-NLP likely to be used:

class PolarityPredictorBowNB(PolarityPredictorWithBagOfWords):

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train NB
        self.model = MultinomialNB()
        self.model.fit(tr_features, tr_targets)

    def predict(
        self, data, get_accuracy = False,
        get_confusion_matrix = False
    ):
        features = self.extract_features(data)
        # use numpy to get predictions
        y_pred = self.model.predict(features)
        # restore labels
        labels = []
        for is_positive in y_pred:
            if is_positive:
                labels.append('pos')
            else:
                labels.append('neg')
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
            return retval
        else:
            return labels



