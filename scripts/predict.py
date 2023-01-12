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
from detector_with_ngrams import SexismDetectorWithNgrams
from tokeniser import simple_tokeniser


def get_seed(args):
    if True:
        # seeding should make no difference in inference mode but let's test this
        # --> keep PRNGs with non-reproducible system seed
        #     and derive a string seed for our functions
        #     that use hashing to randomise data
        import base64
        import numpy
        # get 256 bit of randomness
        seed = base64.b64encode(numpy.random.bytes(32)).decode('utf-8')
    return seed

def get_test_data(args, seed):
    return EDOSDataset(
        seed,
        args.dataset_path, args.run, args.settype,
        unit = 'document', items_per_unit = 1,
        fraction_of_subunits = None,
        number_of_subunits = None,
        deduplicate = False,
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a EDOS sexism detection model")
    parser.add_argument(
            '--dataset_path', type=str, default='data',
            help='Where to find the EDOS data (default: data)',
            )
    parser.add_argument(
            '--model', type=str, default='model-for-task-a.out',
            help='Read the model from this file'
                 ' (default: model-for-task-a.out)',
            )
    parser.add_argument(
            '--run',  type=int, default=1,
            help='Cross-validation run, e.g. 1 to 5 for 5-fold;'
                 ' ignored when testing on an official dev or test set'
                 ' (default: 1)',
            )
    parser.add_argument(
            '--settype', type=str, default='internal-dev',
            help='Which test set to use: internal-dev, official-dev-a, b, c or'
                 ' official-test; in case of "internal-dev", --run specifies'
                 ' specifies which cross-validation fold is used'
                 ' (default: internal-dev)',
            )
    print('Parsing arguments...')
    args = parser.parse_args()
    print('Seeding PRNGs...')
    seed = get_seed(args)
    print('Dateset seed:', seed)
    test_data = get_test_data(args, seed)
    print('Number of test items:', len(test_data))
    print('Loading model...')
    detector = joblib.load(args.model)
    print('Making predictions...')
    predictions = detector.predict(test_data)
    print('Made %d prediction(s)' %len(predictions))
    # TODO: write or print predictions in EDOS format

if __name__ == '__main__':
    main()
