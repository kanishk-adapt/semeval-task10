#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from collections import defaultdict
import sys
import random

def set_seed(args):
    if args.seed:
        import numpy
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
            '--stochastic', action='store_true',
            help='Choose the label at random with probabilities according to votes'
                 ' (default: choose majority label)',
            )
    parser.set_defaults(stochastic=False)
    parser.add_argument(
            '--least_votes', action='store_true',
            help='Choose the label that receives the least number of non-zero votes'
                 ' (default: choose the label that receives the highest number of votes)',
            )
    parser.set_defaults(least_votes=False)
    parser.add_argument(
            '--debug_rows', type=str, default='',
            help='Comma-separated list of rows for which to print debug information'
                 ' (default: empty string = no debug output)',
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
    max_weight = sum(weights)
    if args.debug_rows:
        args.debug_rows = list(map(int, args.debug_rows.split(',')))
        print('Debugging rows', args.debug_rows)
    print('Seeding PRNGs...')
    set_seed(args)
    print('Reading predictions...')
    # read files
    doc_id2predictions = {}
    documents = []  # keep track of order
    debug_documents = []
    for input_index, input_path in enumerate(args.input):
        weight = weights[input_index]
        f = open(input_path, 'rt')
        header = f.readline()
        assert header.rstrip() == 'rewire_id,label_pred'
        row_index = 1
        while True:
            line = f.readline()
            if not line:
                break
            fields = line.rstrip().split(',')
            doc_id = fields[0]
            prediction = ','.join(fields[1:])  # some labels contain a comma
            if doc_id not in doc_id2predictions:
                doc_id2predictions[doc_id] = []
                documents.append(doc_id)
                if args.debug_rows and row_index in args.debug_rows:
                    print('Row', row_index, 'with new document', doc_id)
                    debug_documents.append(doc_id)
            doc_id2predictions[doc_id].append((
                weight, prediction
            ))
            if args.debug_rows and row_index in args.debug_rows:
                print('Updated document', doc_id, 'for row', row_index, 'to')
                for weight, prediction in doc_id2predictions[doc_id]:
                    print('\tweight', weight, 'for', prediction)
            row_index += 1
        f.close()
    print('Number of input predictions:', n_inputs)
    # print predictions in EDOS format
    if '%' in args.output:
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
        debug = doc_id in debug_documents
        label2votes = defaultdict(lambda: 0.0)
        total_weight = 0.0
        for weight, prediction in doc_id2predictions[doc_id]:
            label2votes[prediction] += weight
            if args.stochastic:
                total_weight += weight
        if debug: print('Voting for', doc_id)
        if len(label2votes) == 1:
            # unanimous prediction, no need to spend time in one of the branches below
            if debug: print('\tunanimous')
            prediction = list(label2votes.keys())[0]
        if args.stochastic:
            # normalise probabilities
            for label in label2votes:
                label2votes[label] /= total_weight
            # invert probabilities if using --least_votes
            for label in label2votes:
                label2votes[label] = 1.0 - label2votes[label]
            # pick label according to probabilities
            labels = []
            weights = []
            for label in label2votes:
                labels.append(label)
                weights.append(label2votes[label])
            prediction = random.choices(labels, weights = weights, k = 1)[0]
        else:
            # find label with highest votes
            if args.least_votes:
                best = (2*max_weight, 0.0, 'n/a')
            else:
                best = (0.0, 0.0, 'n/a')
            for label in label2votes:
                votes = label2votes[label]
                if debug: print('\tLabel', label, 'with', votes, 'votes')
                candidate = (votes, random.random(), label)
                if candidate > best and not args.least_votes \
                or candidate < best and args.least_votes:
                    best = candidate
                    if debug: print('\tUpdated best to', best)
                elif debug: print('\tIgnored inferior candidate', candidate)
            prediction = best[2]
        if debug: print('Prediction for', doc_id, 'is:', prediction)
        # note that the prediction is already in csv format
        print('%s,%s' %(doc_id, prediction), file=out)
    if args.output != '-':
        out.close()

if __name__ == '__main__':
    main()
