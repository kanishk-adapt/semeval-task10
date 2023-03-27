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

def get_test_data(args, seed, task = 'a', load_tags = True):
    return EDOSDataset(
        seed,
        args.dataset_path, args.run, args.settype,
        unit = 'document', items_per_unit = 1,
        fraction_of_subunits = None,
        number_of_subunits = None,
        deduplicate = False,
        skip_not_sexist = False if task == 'a' else True,
        load_tags = load_tags,
    )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a EDOS sexism detection model")
    parser.add_argument(
            '--dataset_path', type=str, default='data',
            help='Where to find the EDOS data (default: data)',
            )
    parser.add_argument(
            '--model', type=str, default='',
            help='Read the model from this file'
                 ' (default: model-for-task-$EDOS_TASK.out)',
            )
    parser.add_argument(
            '--output', type=str, default='predictions-%(task)s-%(settype)s-%(rnd)x.csv',
            help='Write predictions in EDOS format to this file;'
                 ' %%(name)format can be used to access local variables;'
                 ' (default: predictions-%%(task)s-%%(settype)s-%%(rnd)x.csv)',
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
    if not args.model:
        import os
        try:
            task = os.environ['EDOS_TASK'].lower()
        except KeyError:
            task = 'a'
        args.model = 'model-for-task-%s.out' %task
    else:
        task = None
    #print('Seeding PRNGs...')
    seed = get_seed(args)
    print('Loading model...')
    detector = joblib.load(args.model)
    print('Dateset seed:', seed)
    test_data = get_test_data(
        args, seed,
        task      = detector.task,
        load_tags = detector.tag_combinations,
    )
    print('Number of test items:', len(test_data))
    print('Making predictions...')
    predictions = detector.predict(test_data)
    print('Made %d prediction(s)' %len(predictions))
    # print predictions in EDOS format
    if task is None:
        task = detector.task
    else:
        assert task == detector.task
    settype = args.settype
    if '%' in args.output:
        import random
        rnd = random.randrange(2**64)
        args.output = args.output %locals()
    if args.output == '-':
        out = sys.stdout
        print('EDOS cvs data follows')
    else:
        out = open(args.output, 'wt')
        print('Writing predictions to', args.output)
    print('rewire_id,label_pred', file=out)
    for index, item in enumerate(test_data):
        assert len(item.documents) == 1  # no subunit sampling for test sets
        doc_idx = item.documents[0]
        assert doc_idx == index  # this should be true for test sets
        doc_id = item.dataset.docs[doc_idx]
        prediction = predictions[index]
        if '"' in prediction or ',' in prediction:
            # cell protection in csv format
            if '"' in prediction:
                prediction = prediction.replace('"', '""')
            prediction = '"%s"' %prediction
        print('%s,%s' %(doc_id, prediction), file=out)
    if args.output != '-':
        out.close()

if __name__ == '__main__':
    main()
