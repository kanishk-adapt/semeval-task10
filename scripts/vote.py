#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from collections import defaultdict
import sys

def set_seed(args):
    if args.seed:
        import numpy
        import random
        numpy.random.seed(args.seed)
        random.seed(args.seed)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train a EDOS sexism detection model")
    parser.add_argument(
            '--output', type=str, default='predictions-v-%(rnd)x.csv',
            help='Write predictions in EDOS format to this file;'
                 ' %%(name)format can be used to access local variables;'
                 ' (default: predictions-v-%%(rnd)x.csv)',
            )
    parser.add_argument(
            '--seed', type=int, default=101,
            help='Initialisation for PRNG (integer), 0 = use system seed'
                 ' (default: 101)',
            )
    parser.add_argument(
            '--task', type=str, default='a',
            help='Which task to train for; one of "a", "b" or "c" (default: a)',
            )
    parser.add_argument(
            '--weights', type=str, default='',
            help='Comman-separated list of weights for each input predictions'
                 ' (default: empty string = uniform weigths)',
            )
    parser.add_argument(
            'input', nargs='+',
            help='Input predictions in EDOS format'
            )
    print('Parsing arguments...')
    args = parser.parse_args()
    n_inputs = len(args.input)
    if args.weights:
        weights = list(map(float, args.weights.split(',')))
        assert len(weights) == n_inputs
    else:
        weights = n_inputs * [1.0]
    print('Seeding PRNGs...')
    set_seed(args)
    print('Reading predictions...')
    # read files
    doc_id2predictions = {}
    documents = []  # keep track of order
    for input_index, input_path in enumerate(args.input):
        weight = weights[input_index]
        f = open(input_path, 'rt')
        header = f.readline()
        assert header.startswith('rewire_id,label_pred')
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.rstrip().split(',')
            doc_id = fields[0]
            prediction = fields[1]
            if doc_id not in doc_id2predictions:
                doc_id2predictions[doc_id] = []
                documents.append(doc_id)
            doc_id2predictions[doc_id].append((
                weight, prediction
            ))
        f.close()
    print('Number of input predictions:', n_inputs)
    # print predictions in EDOS format
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
    if args.task == 'a':
        print('rewire_id,label_pred', file=out)
    else:
        raise NotImplementedError
    for doc_id in documents:
        label2votes = defaultdict(lambda: 0.0)
        for weight, prediction in doc_id2predictions[doc_id]:
            label2votes[prediction] += weight
        # find label with highest votes
        best = (0.0, 0.0, 'n/a')
        for label in label2votes:
            votes = label2votes[label]
            candidate = (votes, random.random(), label)
            if candidate > best:
                best = candidate
        prediction = best[2]
        print('%s,%s' %(doc_id, prediction), file=out)
    if args.output != '-':
        out.close()

if __name__ == '__main__':
    main()
