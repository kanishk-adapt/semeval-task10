#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from collections import defaultdict
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import sys
from xgboost import XGBClassifier

from basic_dataset import Concat
from edos_dataset import EDOSDataset
from detector_with_ngrams import SexismDetectorWithNgramsAndWordlists
from tokeniser import simple_tokeniser, nltk_tokeniser, spacy_tokeniser, \
                      gensim_tokeniser


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
    if args.settype in ('internal', 'official'):
        args.settype = args.settype + '-training'
    training_data = []
    # allow both underscore and minus in --augmentation
    args.augmentation = args.augmentation.replace('_', '-')
    if 'exclude-basic' not in args.augmentation:
        # usually, we include a copy of the training data as is, i.e.
        # 1 copy of each document with no sampling of sub-units,
        # unless the user specifies "exclude-basic"
        training_data.append(EDOSDataset(
            'basic:' + seed,
            args.dataset_path, args.run, args.settype,
            unit = 'document', items_per_unit = 1,
            fraction_of_subunits = None,
            number_of_subunits = None,
            deduplicate = True,
            skip_not_sexist = args.task in ('b', 'c'),
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
                args.dataset_path, args.run, args.settype,
                unit = 'class', items_per_unit = int(fields[0]),
                fraction_of_subunits = None,
                number_of_subunits = int(fields[4]),
                deduplicate = True,
                skip_not_sexist = args.task in ('b', 'c'),
                simplify_labels_for_augmentation = args.task if args.simplify_labels_for_augmentation else None,
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
    if args.classifier == 'MultinomialNB':
        return MultinomialNB(fit_prior = not args.uniform_prior)
    elif args.classifier == 'ComplementNB':
        return ComplementNB(fit_prior = not args.uniform_prior)
    elif args.classifier == 'BernoulliNB':
        return BernoulliNB(fit_prior = not args.uniform_prior)
    elif args.classifier == 'DecisionTree':
        return DecisionTreeClassifier()
    elif args.classifier == 'DecisionTreeM08':
        return DecisionTreeClassifier(min_samples_leaf = 8)
    elif args.classifier == 'DecisionTreeM10':
        return DecisionTreeClassifier(min_samples_leaf = 10)
    elif args.classifier == 'DecisionTreeM15':
        return DecisionTreeClassifier(min_samples_leaf = 15)
    elif args.classifier == 'DecisionTreeM50':
        return DecisionTreeClassifier(min_samples_leaf = 50)
    elif args.classifier == 'RandomForest':
        return RandomForestClassifier()
    elif args.classifier == 'RandomForestM10':  # TODO: add option to control min_samples_leaf
        return RandomForestClassifier(min_samples_leaf = 10)
    elif args.classifier == 'RandomForestM50':
        return RandomForestClassifier(min_samples_leaf = 50)
    elif args.classifier in ('LinearSVM', 'LinearSVC'):
        return LinearSVC(C=args.c)
    elif args.classifier == 'XGBoost':
        return XGBClassifier(use_label_encoder=False)  # our labels are encoded already
    else:
        raise ValueError('unknown classifier %s' %args.classifier)

def get_tokeniser(args):
    if args.tokeniser == 'simple':
        return simple_tokeniser  # cannot use lambda here as lambda functions cannot be pickled without some extra tricks
    elif args.tokeniser == 'nltk':
        return nltk_tokeniser
    elif args.tokeniser == 'spacy':
        return spacy_tokeniser
    elif args.tokeniser == 'gensim':
        return gensim_tokeniser
    else:
        raise ValueError('unknown tokeniser %s' %args.tokeniser)

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
            '--augmentation', type=str, default='100-per-class-with-3-documents',
            help='How to augment the training data;'
                 ' space-separated list of methods;'
                 ' include "exclude_basic" to not include'
                 ' a copy of the basic dataset;'
                 ' none = no data augmentation'
                 ' (default: 100-per-class-with-3-documents)',
            )
    parser.add_argument(
            '--simplify_labels_for_augmentation', dest='simplify_labels_for_augmentation', action='store_true',
            help='When sampling documents in concatenation-based data augmentation,'
                 ' group documents according to the label of the target task'
                 ' (default: use the full task C label for grouping)'
            )
    parser.set_defaults(simplify_labels_for_augmentation=False)
    parser.add_argument(
            '--task', type=str, default='',
            # default is set after checking $EDOS_TASK below
            help='Which task to train for; one of "a", "b" or "c"'
                 ' (default: use environment variable EDOS_TASK and'
                 ' fall back to task "a" if not set)',
            )
    parser.add_argument(
            '--write_model_to', type=str, default='model-for-task-%s.out',
            help='Where to write the model file '
                 ' (default: model-for-task-%%s.out where %%s is replaced with the task code)',
            )
    parser.add_argument(
            '--write_features_to', type=str, default='',
            help='Whether and where to write the feature matrix '
                 ' (default: do not write features to disk)',
            )
    parser.add_argument(
            '--write_column_names_to', type=str, default='',
            help='Whether and where to write the column names of the feature matrix '
                 ' (default: do not write column names to disk)',
            )
    parser.add_argument(
            '--write_data_to', type=str, default='',
            help='Whether and where to write the training data, including '
                 ' synthetic data if data augmentation is used'
                 ' (default: do not write training data to disk)',
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
                 ' and the run (see --run) specifies which part;'
                 ' can also use dev and test sets,'
                 ' e.g. for feature extraction'
                 ' (default: internal)',
            )
    parser.add_argument(
            '--tokeniser', type=str, default='gensim',
            help='What tokeniser to use; one of simple, nltk, spacy or gensim'
                 ' (default: gensim)',
            )
    parser.add_argument(
            '--no_lowercase_ngrams', dest='use_lowercase', action='store_false',
            help='Do not create lowercase ngram features if --ngrams is not empty'
                 ' (default is to create both truecase and lowercase ngrams);'
                 ' see also --no-truecase-ngrams',
            )
    parser.set_defaults(use_lowercase=True)
    parser.add_argument(
            '--no_truecase_ngrams', dest='use_truecase', action='store_false',
            help='Do not create truecase ngram features if --ngrams is not empty'
                 ' (default is to create both truecase and lowercase ngrams);'
                 ' see also --no-lowercase-ngrams',
            )
    parser.set_defaults(use_truecase=True)
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
            '--tag_combinations', type=str, default='p,s',
            help='Create ngram features from tag combinations for each'
                 ' comma-separted combination of tags (t=token, p=POS,'
                 ' d=dependency relation, s=sentiment); multiple tags'
                 ' can be concatenated to form complex tags, e.g. "ps"'
                 ' may produce a tag "NOUN neutral"'
                 ' (default: p,s)',
            )
    parser.add_argument(
            '--wordlist_folder', type=str, default='data/wordlists',
            help='Where to find wordlists; only files with .txt suffix will be'
                 ' read; if the filename (not the path) contains "-truecase",'
                 ' word matching will be case-sentive'
                 ' (default: data/wordlists)',
            )
    parser.add_argument(
            '--min_freq', type=int, default=14,
            help='Events must have at least this frequency to be included'
                 ' in the vocabulary (default: 14)',
            )
    parser.add_argument(
            '--clip_counts', type=float, default=1.0,
            help='Event counts are raised to the power of (1-CLIP_COUNT) to'
                 ' obtain feature values; 0 = use counts as is; 1 = use binary'
                 ' indicator features; 0.6667 = use cubic root'
                 ' (default: 1 = binary features)',
            )
    parser.add_argument(
            '--normalise_by_number_of_documents', type=str, default='after-clipping',
            help='Normalise features that are based on event counts by'
                 ' the number of documents in the training or test item;'
                 ' one of "no" (be agnostic to the number of documents),'
                 ' "before-clipping" and "after-clipping"'
                 ' (default: after-clipping)',
            )
    parser.add_argument(
            '--classifier', type=str, default='LinearSVM',
            help='What classifier to use. One of'
                 ' BernoulliNB,'
                 ' ComplementNB,'
                 ' DecisionTree,'
                 ' DecisionTreeM08 (min_samples_leaf=8),'
                 ' DecisionTreeM10 (min_samples_leaf=10),'
                 ' DecisionTreeM15 (min_samples_leaf=15),'
                 ' DecisionTreeM50 (min_samples_leaf=50),'
                 ' LinearSVM,'
                 ' MultinomialNB,'
                 ' RandomForest,'
                 ' RandomForestM10 (min_samples_leaf=10),'
                 ' RandomForestM50 (min_samples_leaf=50),'
                 ' RandomForest,'
                 ' XGBoost'
                 ' (default: LinearSVM)',
            )
    parser.add_argument(
            '--c', type=float, default='0.025',
            help='Parameter C for LinearSVM'
                 ' (default: 0.025)',
            )
    parser.add_argument(
            '--uniform_prior', action='store_true',
            help='Use uniform prior with NaiveBayes'
                 ' (default is to fit prior to training data)',
            )
    parser.set_defaults(uniform_priors=False)

    print('Parsing arguments...')
    args = parser.parse_args()
    if not args.task:
        import os
        try:
            args.task = os.environ['EDOS_TASK'].lower()
            print('Setting task to "%s" as per $EDOS_TASK' %args.task)
        except KeyError:
            args.task = 'a'
            print('Using default task "%s"' %args.task)
    assert args.task in ('a', 'b', 'c')
    print('Seeding PRNGs...')
    seed = get_seed(args)
    print('Dateset seed:', seed)
    training_data = get_training_data(args, seed)
    if args.write_data_to:
        # write training data to disk
        print('Saving training data to %s...' %args.write_data_to)
        f = open(args.write_data_to, 'wt')
        training_data.save_to_file(f, fileformat = 'edos')
        f.close()
    print('Number of training items:', len(training_data))
    args.tag_combinations = args.tag_combinations.replace(' ', ',')
    args.tag_combinations = args.tag_combinations.replace('+', '')
    detector = SexismDetectorWithNgramsAndWordlists(
            task  = args.task,
            model = get_internal_model(args),
            tokeniser = get_tokeniser(args),
            min_freq  = args.min_freq,
            ngram_range = get_ngram_range(args),
            padding     = get_padding(args),
            use_truecase = args.use_truecase,
            use_lowercase = args.use_lowercase,
            tag_combinations = args.tag_combinations,
            clip_counts = args.clip_counts,
            normalise_by_number_of_documents = args.normalise_by_number_of_documents,
            wordlist_folder = args.wordlist_folder
    )
    print('Training...')
    features = detector.train(
        training_data,
        get_features = args.write_features_to
    )
    if args.write_features_to:
        # write features to disk
        print('Saving features to %s...' %args.write_features_to)
        joblib.dump(features, args.write_features_to)
    if args.write_column_names_to:
         # write column names to disk
         print('Saving column names to %s...' %args.write_column_names_to)
         f = open(args.write_column_names_to, 'wt')
         f.write('feature_matrix_column_name\n')
         for col_name in detector.get_feature_matrix_column_names():
             f.write(col_name)
             f.write('\n')
         f.close()
    # write model to disk
    if '%s' in args.write_model_to:
        path = args.write_model_to %args.task
    elif not args.write_model_to:
        # user set output name to the empty string
        # --> user doesn't want to save the model
        sys.exit(0)
    else:
        path = args.write_model_to
    print('Saving model to %s...' %path)
    # We save the detector as the model doesn't know how
    # to map text to features and label indices to labels
    joblib.dump(detector, path)
    if '-dev' in args.settype or '-test' in args.settype:
        print('Warning: model has been trained on dev or test data')

if __name__ == '__main__':
    main()
