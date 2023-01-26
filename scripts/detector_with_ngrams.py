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
import numpy
import os
from sklearn.naive_bayes import MultinomialNB
import sys
from tqdm import tqdm

from detector_base import SexismDetector


fine_grained_labels = '1.1 1.2 2.1 2.2 2.3 3.1 3.2 3.3 3.4 4.1 4.2'.split()

label2index = {}
for index, label in enumerate(fine_grained_labels):
    label2index[label] = index


class SexismDetectorWithVocab(SexismDetector):

    def __init__(self,
        tokeniser = None, min_freq = 5, clip_counts = 1,
        normalise_by_number_of_documents = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokeniser = tokeniser
        self.min_freq  = min_freq
        self.clip_counts = clip_counts
        self.normalise_by_number_of_documents = normalise_by_number_of_documents

    def train(self, data_with_labels, **kwargs):
        # (1) build the vocabulary from the training data
        self.reset_vocab()
        self.add_to_vocab_from_data(data_with_labels)
        self.finalise_vocab()
        # (2) extract features and train the model
        return super().train(data_with_labels, **kwargs)

    def reset_vocab(self):
        self.vocab = defaultdict(lambda: 0)   # for each entry, record number of occurrences

    def add_to_vocab_from_data(self, data):
        ''' expand vocabulary to cover new data '''
        for item in data:
            for event in self.get_item_events(item):
                self.vocab[event] += 1

    def finalise_vocab(self):
        ''' finish creating the vocaulary and create support structures '''
        # apply frequency threshold and reduce to sorted list
        selected_vocab = []
        for entry in self.vocab:
            if self.vocab[entry] >= self.min_freq:
                selected_vocab.append(entry)
        self.vocab = sorted(selected_vocab)
        # create reverse map for fast token lookup
        self.event2index = {}
        for index, event in enumerate(self.vocab):
            self.event2index[event] = index

    def get_vector_length(self):  # sub-classes may want to add components for non-vocab features
        return len(self.vocab)

    def get_item_feature_vector(self, item):  # sub-classes may want to add features here
        columns = self.get_vector_length()
        dtype   = self.get_vector_dtype()
        vector = numpy.zeros((columns,), dtype=dtype)
        non_zero_columns = set()
        for event in self.get_item_events(item):
            try:
                index = self.event2index[event]
            except KeyError:  # token not in vocab
                continue      # --> skip this event
            if self.clip_counts == 1.0:
                vector[index] = 1
            else:
                vector[index] += 1
            non_zero_columns.add(index)
        if self.clip_counts > 0.0 and self.clip_counts < 1.0:
            # map values in such a way that self.clip_counts close to 0
            # behaves similarly to self.clip_counts == 0 and
            # self.clip_counts close to 1 behaves similarly to
            # self.clip_counts == 1
            #for column in range(columns):  # TODO: usually, there are only a few non-zero columns
            for column in non_zero_columns:
                vector[index] = vector[index] ** (1.0 - self.clip_counts)
        if self.normalise_by_number_of_documents:
            n_docs = len(item.documents)
            if n_docs > 1:
                for column in non_zero_columns:
                    vector[index] = vector[index] / float(n_docs)
        return vector

    def get_feature_matrix_column_names(self):
        # self.vocab determines the position of each feature in
        # get_item_feature_vector() above
        # --> the list of features is self.vocab
        return self.vocab

    # the following functions will have to be implemented in sub-classes
    # to be able to use above functionality

    def get_item_events(self, item):
        raise NotImplementedError


class SexismDetectorWithNgrams(SexismDetectorWithVocab):

    def __init__(self,
        ngram_range = None, padding = None,
        use_truecase = True, use_lowercase = True,
        tag_combinations = 'p,d,s,tp,pd',
        **kwargs
    ):
        super().__init__(**kwargs)
        if not ngram_range:
            ngram_range = [1]
        self.ngram_range = ngram_range
        self.padding = padding
        self.use_truecase = True
        self.use_lowercase = True
        self.tag_combinations = []
        for tag_combination in tag_combinations.split(','):
            # expand e.g. 'pd' to ('p', 'd')
            self.tag_combinations.append(tuple(tag_combination))

    def get_item_events(self, item):
        for tag_name, sequence in self.get_item_sequences(item):
            for event in self.get_squence_events(sequence, tag_name):
                yield event

    def get_squence_events(self, item_tokens, tag_name):
        for n in self.ngram_range:
            assert n > 0
            tokens = item_tokens[:]
            if self.padding and n > 1:
                n1_padding = (n-1) * [self.padding]
                tokens = n1_padding + tokens + n1_padding
            seq_length = len(tokens)
            start = 0
            while start < seq_length + 1 - n:
                ngram = ['%s:%d' %(tag_name, n)] + tokens[start:start+n]
                yield ' '.join(ngram)
                start += 1

    def get_item_sequences(self, item):
        if self.use_truecase:
            yield ('TT', self.tokeniser(item.get_text()))
        if self.use_lowercase:
            yield ('TL', self.tokeniser(item.get_text().lower()))
        if item.dataset.tags and self.tag_combinations:
            for tag_combination in self.tag_combinations:
                tag_combo_name = '+'.join(tag_combination)
                sequence = item.get_tags(tag_combination)
                yield (tag_combo_name, sequence)
        elif self.tag_combinations:
            raise ValueError('Requested tag-based features but tags not loaded')


class SexismDetectorWithNgramsAndWordlists(SexismDetectorWithNgrams):

    def __init__(self,
        wordlist_folder = 'data/wordlists',
        **kwargs
    ):
        super().__init__(**kwargs)
        relevant_filenames = []
        for filename in os.listdir(wordlist_folder):
            if filename.endswith('.txt'):
                relevant_filenames.append(filename)
        self.wordlists = []
        for filename in tqdm(relevant_filenames, desc='Loading wordlist(s)'):
           f = open(os.path.join(wordlist_folder, filename), 'rt')
           is_lower = '-truecase' not in filename
           words = set()
           while True:
               line = f.readline()
               if not line:
                   break
               if is_lower:
                   assert line == line.lower()
               for word in line.split():
                   words.add(word)
           f.close()
           self.wordlists.append((filename[:-4], is_lower, words))

    def get_item_events(self, item):
        for event in super().get_item_events(item):
            yield event
        if not self.wordlists:
            return
        text = item.get_text()
        tokens_truecase = None
        tokens_lowercase = None
        for wl_name, wl_is_lower, wl_words in self.wordlists:
            if wl_is_lower:
                if tokens_lowercase is None:
                    tokens_lowercase = self.tokeniser(text.lower())
                tokens = tokens_lowercase
            else:
                if tokens_truecase is None:
                    tokens_truecase = self.tokeniser(text)
                tokens = tokens_truecase
            for token in tokens:
                if token in wl_words:
                    yield 'WL:' + wl_name
