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

from predictor_interface import SexismDetector


fine_grained_labels = '1.1 1.2 2.1 2.2 2.3 3.1 3.2 3.3 3.4 4.1 4.2'.split()

label2index = {}
for index, label in enumerate(fine_grained_labels):
    label2index[label] = index


class SexismDetectorWithVocab(SexismDetector):

    def __init__(self, tokeniser = None, min_freq = 5, **kwargs)
        super.__init__(self, **kwargs)
        self.tokeniser = tokeniser
        self.min_freq  = min_freq

    def train(self, data_with_labels):
        # (1) build the vocabulary from the training data
        self.reset_vocab()
        self.add_to_vocab_from_data(data_with_labels)
        self.finalise_vocab()
        # (2) extract features and train the model
        super.train(data_with_labels)
        
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

    # the following functions will have to be implemented in sub-classes
    # to be able to use above functionality

    def get_item_atoms(self, item):
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

