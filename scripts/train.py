#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from sklearn.naive_bayes import MultinomialNB
import sys

from basic_dataset import Concat
from edos_dataset import EDOSDataset
from predictor_with_ngrams import SexismDetectorWithNgrams

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
            '--data_augmentation', type=str, default='none',
            help='How to augment the training data;'
                 ' space-separated list of methods;'
                 ' include "exclude_basic" to not include'
                 ' a copy of the basic dataset;'
                 ' (default: none = no augmentation)',
            )
    parser.add_argument(
            '--task', type=str, default='a',
            help='Which task to train for; one of "a", "b" or "c" (default: a)',
            )
    parser.add_argument(
            '--write_model_to', type=str, default='model-for-task-X.out',
            help='Where to write the model file '
                 ' (default: model-for-task-X.out where X is replaced with the task code)',
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
    print('Parsing arguments...')
    args = parser.parse_args()
    print('Seeding PRNGs...')
    if args.seed:
        import numpy
        import random
        numpy.random.seed(args.seed)
        random.seed(args.seed)
        seed = '%d' %args.seed
    else:
        import base64
        import numpy
        seed = base64.b64encode(numpy.random.bytes(32)).decode('utf-8')
    print('Dateset seed:', seed)
    training_data = []
    # allow both underscore and minus in --data-augmentation
    args.data_augmentation = args.data_augmentation.replace('_', '-')
    if 'exclude-basic' not in args.data_augmentation:
        # usually, we include a copy of the training data as is, i.e.
        # 1 copy of each document with no sampling of sub-units
        training_data.append(EDOSDataset(
            'basic:' + seed,
            args.dataset_path, args.run, args.settype + '-training',
            unit = 'document', items_per_unit = 1,
            fraction_of_subunits = None,
            number_of_subunits = None,
            deduplicate = True,
        ))
    # support alternative delimiters in --data-augmentation
    for delimiter in ',| ':
        args.data_augmentation = args.data_augmentation.replace(delimiter, '+')
    for augmentation in args.data_augmentation.split('+'):
        if augmentation in ('none', 'exclude-basic'):
            continue
        if 'per-class-with' in augmentation:  # e.g. 100-per-class-with-3-documents
            fields = augmentation.split('-')
            # [0] [1] [2]   [3] [4] [5]
            # 100 per class with 3 documents
            assert len(fields) == 6
            assert fields[5] == 'documents'
            training_data.append(EDOSDataset(
                augmentation + ':' + seed,
                args.dataset_path, args.run, args.settype + '-training',
                unit = 'class', items_per_unit = int(fields[0]),
                fraction_of_subunits = None,
                number_of_subunits = int(fields[4]),
                deduplicate = True,
            ))
        else:
            raise ValueError('unknown augmentation profile %s' %augmentation)
    if len(training_data) == 1:
        training_data = training_data[0]
    else:
        training_data = Concat(training_data)

    print('Number of training items:', len(training_data))

if __name__ == '__main__':
    main()
