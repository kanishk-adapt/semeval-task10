#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from collections import defaultdict
import joblib
from sklearn.naive_bayes import MultinomialNB
import sys

from basic_dataset import Concat
from edos_dataset import EDOSDataset
from predictor_with_ngrams import SexismDetectorWithNgrams

def get_seed(args):
    if args.seed:
        import numpy
        import random
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        seed = '%d' %args.seed
    else:
        # user wants to use non-reproducible system seed
        # --> need to derive a string seed for our functions
        #     that use hashing to randomise data
        import base64
        import numpy
        # get 256 bit of randomness
        seed = base64.b64encode(numpy.random.bytes(32)).decode('utf-8')
    return seed

def get_training_data(args, seed):
    training_data = []
    # allow both underscore and minus in --augmentation
    args.augmentation = args.augmentation.replace('_', '-')
    if 'exclude-basic' not in args.augmentation:
        # usually, we include a copy of the training data as is, i.e.
        # 1 copy of each document with no sampling of sub-units,
        # unless the user specifies "exclude-basic"
        training_data.append(EDOSDataset(
            'basic:' + seed,
            args.dataset_path, args.run, args.settype + '-training',
            unit = 'document', items_per_unit = 1,
            fraction_of_subunits = None,
            number_of_subunits = None,
            deduplicate = True,
        ))
    # support alternative delimiters in --augmentation
    for delimiter in ',| ':
        args.augmentation = args.augmentation.replace(delimiter, '+')
    # process the list of requested data augmentations
    aug2count = defaultdict(lambda: 0)
    for augmentation in args.augmentation.split('+'):
        if augmentation in ('none', 'exclude-basic'):
            # no data to add
            continue
        # in case the same augmentation is used multiple times, we
        # need to know this to use a different random seed below
        aug2count[augmentation] += 1
        a_seq = aug2count[augmentation]
        if 'per-class-with' in augmentation:  # e.g. 100-per-class-with-3-documents
            fields = augmentation.split('-')
            # [0] [1] [2]   [3] [4] [5]
            # 100 per class with 3 documents
            assert len(fields) == 6
            assert fields[5] == 'documents'
            training_data.append(EDOSDataset(
                '%s:%d:%s' %(augmentation, a_seq, seed),
                args.dataset_path, args.run, args.settype + '-training',
                unit = 'class', items_per_unit = int(fields[0]),
                fraction_of_subunits = None,
                number_of_subunits = int(fields[4]),
                deduplicate = True,
            ))
        elif augmentation.endswith('-sample-with-replacement'):
            fields = augmentation.split('-')
            assert len(fields) == 4
            target_size = float(fields[0])
            # TODO: sample with replacement
            raise NotImplementedError
        elif augmentation.endswith('-sample-without-replacement'):
            fields = augmentation.split('-')
            assert len(fields) == 4
            target_size = float(fields[0])
            if target_size >= 1.0:
                copies = int(target_size)  # rounding down
                # TODO: append full copies of data
                raise NotImplementedError
                target_size -= copies
            if target_size > 0.0:
                # TODO: create sample and append to data
                raise NotImplementedError
        else:
            raise ValueError('unknown augmentation %s' %augmentation)
    if len(training_data) == 1:
        training_data = training_data[0]
    elif len(training_data) == 0:
        raise ValueError('When using exclude-basic, you must add at least one augmentation method')
    else:
        training_data = Concat(training_data)
    return training_data

def get_internal_model(args):
    # TODO: support other choices via args
    return MultinomialNB()

def simple_tokeniser(text):
    return text.split()

def get_tokeniser(args):
    return simple_tokeniser  # cannot use lambda here as lambda functions cannot be pickled without some extra tricks

def get_ngram_range(args):
    if not args.ngrams:
        return None  # default: unigrams only
    for delimiter in ',| ':  # alternative delimiters
        args.ngrams = args.ngrams.replace(delimiter, '+')
    values = sorted(set(map(int, args.ngrams.split('+'))))
    return tuple(values)

def get_padding(args):
    if not args.padding or args.padding.lower() == 'none':
        return None
    return args.padding

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a EDOS sexism detection model")
    parser.add_argument(
            '--seed', type=int, default=101,
            help='Initialisation for PRNG (integer), 0 = use system seed'
                 ' (default: 101)',
            )
    parser.add_argument(
            '--dataset_path', type=str, default='data',
            help='Where to find the EDOS data (default: data)',
            )
    parser.add_argument(
            '--augmentation', type=str, default='none',
            help='How to augment the training data;'
                 ' space-separated list of methods;'
                 ' include "exclude_basic" to not include'
                 ' a copy of the basic dataset;'
                 ' (default: none = no data augmentation)',
            )
    parser.add_argument(
            '--task', type=str, default='a',
            help='Which task to train for; one of "a", "b" or "c" (default: a)',
            )
    parser.add_argument(
            '--write_model_to', type=str, default='model-for-task-%s.out',
            help='Where to write the model file '
                 ' (default: model-for-task-%s.out where %s is replaced with the task code)',
            )
    parser.add_argument(
            '--run',  type=int, default=1,
            help='Cross-validation run, e.g. 1 to 5 for 5-fold;'
                 ' ignored when training on the official training set'
                 ' (default: 1)',
            )
    parser.add_argument(
            '--settype', type=str, default='internal',
            help='Which training set to use: internal or official;'
                 ' "internal" uses 80%% of the official training data'
                 ' and the run (see --run) specifies which part'
                 ' (default: internal)',
            )
    parser.add_argument(
            '--ngrams', type=str, default='1,2,3',
            help='What values of n to use when producing n-grams as text features'
                 ' (default: 1,2,3 = unigrams, bigrams and trigrams)',
             )
    parser.add_argument(
            '--padding', type=str, default='[PAD]',
            help='String to use as padding token; '
                 ' the empty string or the keyword "none" de-activate padding'
                 ' (default: [PAD])',
            )
    parser.add_argument(
            '--min_freq', type=int, default=5,
            help='Events must have at least this frequency to be included'
                 ' in the vocabulary (default: 5)',
            )
    parser.add_argument(
            '--clip_counts', type=float, default=1.0,
            help='Event counts are raised to the power of (1-CLIP_COUNT) to'
                 ' obtain feature values; 0 = use counts as is; 1 = use binary'
                 ' indicator features; 0.6667 = use cubic root'
                 ' (default: 1 = binary features)',
            )
    print('Parsing arguments...')
    args = parser.parse_args()
    print('Seeding PRNGs...')
    seed = get_seed(args)
    print('Dateset seed:', seed)
    training_data = get_training_data(args, seed)
    print('Number of training items:', len(training_data))
    detector = SexismDetectorWithNgrams(
            task  = args.task,
            model = get_internal_model(args),
            tokeniser = get_tokeniser(args),
            min_freq  = args.min_freq,
            ngram_range = get_ngram_range(args),
            padding     = get_padding(args),
    )
    print('Training...')
    detector.train(training_data)
    # write model to disk
    path = args.write_model_to %args.task
    print('Saving model to %s...' %path)
    # We save the detector as the model doesn't know how
    # to map text to features and label indices to labels
    joblib.dump(detector, path)

if __name__ == '__main__':
    main()