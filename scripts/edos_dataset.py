#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2022 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from __future__ import print_function

from basic_dataset import Dataset, Item

import pandas
import os
import sys
from tqdm import tqdm

settype2setting = {
    # dataset name --> path suffix, is_labelled, cross-validation
    'official-dev-a': ('dev-task-a/dev_task_a_entries.csv', False, False),
    'official-dev-b': ('dev-task-b/dev_task_b_entries.csv', False, False),
    'official-dev-c': ('dev-task-c/dev_task_c_entries.csv', False, False),
    # TODO: official test sets
    'official-training': ('train_all_tasks.csv', True, False),
    'internal-training': ('k8020-tr-run-%d.csv', True, True),
    'internal-dev':      ('k8020-dev-run-%d.csv', True, True),
}

class EDOSDataset(Dataset):

    def load_from_path(self,
        path = 'data/edos', run = 1, settype = 'test',
        skip_not_sexist = False,
        **kwargs
    ):
        """ Append data from `path` to the data set.
            `kwargs` can be used to control subset selection via
            self.usable().
            `run` is ignored for settype2setting[settype][-1] == False
        """
        global settype2setting
        self.is_tokenised = False
        path_suffix, is_labelled, cross_validation = settype2setting[settype]
        if cross_validation:
            path_suffix = path_suffix %run
        if skip_not_sexist and not is_labelled:
            raise ValueError('Cannot skip non-sexists items in unlabelled data')
        data_path = os.path.join(path, path_suffix)
        # https://stackoverflow.com/questions/16988526/pandas-reading-csv-as-string-type
        # needed for text "nan" as in "love my nan, mum and sister" not the be mapped
        # to special float NaN (not a number)
        df = pandas.read_csv(data_path, dtype=str, na_filter=False)
        # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Loading data'):  # https://stackoverflow.com/questions/47087741/use-tqdm-progress-bar-with-pandas
            if is_labelled:
                label = row['label_vector'][:4].rstrip()  # 'none' or '%d.%d' format
            else:
                label = 'n/a'
            if skip_not_sexist and label == 'none':
                continue
            document = row['text']
            if not self.usable_document(document, **kwargs):
                continue
            doc_id  = row['rewire_id']
            doc_idx = self.get_doc_idx(doc_id)
            self.documents.append(document)
            if is_labelled:
                self.doc2label.append(label)
                if not label in self.labels:
                    self.labels.append(label)
            if self.debug:
                sys.stderr.write('Loaded document %s with label %s\n' %(doc_id, label))
        self.is_labelled = is_labelled
        # load POS and deprel features
        features_path = os.path.join(path, 'dep-pos', 'extracted_features.csv')
        fast_features_path = os.path.join(  # e.g. extracted_features-k8020-dev-run-1.csv
            path, 'dep-pos', 'extracted_features-%s' %path_suffix
        )
        tag_names = 'token pos_tag dep_tag sentiment'.split()
        if os.path.exists(fast_features_path):
            df = pandas.read_csv(fast_features_path, dtype=str, na_filter=False)
            save_ffp = False
        elif os.path.exists(features_path):
            df = pandas.read_csv(features_path, dtype=str, na_filter=False)
            save_ffp = True
        else:
            df = None
            sys.stderr.write('Warning: dep-pos features not found\n')
            self.tags = None
        if df is not None:
            last_doc_id = None
            doc_tags = {}
            all_tags = []
            for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Loading tags'):
                #print('F row', row)
                doc_id = row['rewire_id']
                if doc_id != last_doc_id and last_doc_id is not None:
                    self.add_doc_tags(all_tags, doc_tags)
                    doc_tags = {}
                if doc_tags:
                    # consistency check
                    assert doc_tags['doc_id'] == doc_id
                else:
                    # initialise doc_tags
                    doc_tags['doc_id'] = doc_id
                    for tag_name in tag_names:
                        doc_tags[tag_name] = []
                # update doc_tags with new row
                for tag_name in tag_names:
                    tag = row[tag_name]
                    try:
                        assert ' ' not in tag
                        #assert ',' not in tag  # actually occurs in token column
                        #assert '"' not in tag
                    except TypeError:
                        raise ValueError('unexpected tag %r of type %s in row %r' %(tag, type(tag), row))
                    except AssertionError:
                        raise ValueError('unexpected tag %r of type %s in row %r' %(tag, type(tag), row))
                    doc_tags[tag_name].append(tag)
                # prepare for next row
                last_doc_id = doc_id
            # complete last item
            if doc_tags:
                self.add_doc_tags(all_tags, doc_tags)
            # check all training items are covered
            # (add_doc_tags() has rejected items outside our
            # dataset, so we only need to check the number
            # of items is correct)
            if len(all_tags) == len(self.documents):
                all_tags.sort()
                assert all_tags[0][0] == 0
                assert all_tags[-1][0] == len(all_tags) - 1
                # strip index numbers from list
                self.tags = list(map(lambda x: x[1], all_tags))
            else:
                sys.stderr.write('Warning: ignoring %s as it does not cover all of %s\n' %(
                    features_path, data_path
                ))
            if save_ffp:
                out = open(fast_features_path, 'wt')
                header = ['rewire_id', 'tag_idx'] + tag_names
                out.write(','.join(header))
                out.write('\n')
                for doc_idx, doc_tags in enumerate(tqdm(self.tags, desc='Saving subset of tags')):
                    n_tags = len(doc_tags[tag_names[0]].split(' '))
                    exp_doc_tags = {}
                    for tag_name in tag_names:
                        exp_doc_tags[tag_name] = doc_tags[tag_name].split(' ')
                        if len(exp_doc_tags[tag_name]) != n_tags:
                            doc_id = self.docs[doc_idx]
                            raise ValueError('mismatch of number of tags for document %s: %d for %s, %d for %s\n' %(
                                doc_id, n_tags, tag_names[0], len(exp_doc_tags[tag_name]), tag_name
                                ))
                    for tag_index in range(n_tags):
                        row = []
                        row.append(self.docs[doc_idx])
                        row.append('%d' %tag_index)
                        for tag_name in tag_names:
                            tag = exp_doc_tags[tag_name][tag_index]
                            if ',' in tag or '"' in tag:  # csv escape
                                tag = tag.replace('"', '""')
                                tag = '"' + tag + '"'
                            row.append(tag)
                        out.write(','.join(row))
                        out.write('\n')
                out.close()

    def add_doc_tags(self, all_tags, doc_tags):
        doc_id = doc_tags['doc_id']
        if not doc_id in self.doc2idx:
            return
        new_doc_tags = {}
        length = None
        for key in doc_tags.keys():
            if key == 'doc_id':
                continue
            tags = doc_tags[key]
            if length is not None:
                assert length == len(tags)
            else:
                length = len(tags)
            new_doc_tags[key] = ' '.join(tags)
        doc_idx = self.doc2idx[doc_id]
        all_tags.append((doc_idx, new_doc_tags))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EDOS data for sexism detection")
    parser.add_argument(
            '--seed', type=str, default='101',
            help='Initialisation for PRNG (string), empty string = use system seed'
                 ' (default: 101)',
            )
    parser.add_argument(
            '--path', type=str, default='data',
            help='Where to find the EDOS data (default: data)',
            )
    parser.add_argument(
            '--run',  type=int, default=1,
            help='Cross-validation run, e.g. 1 to 5 for 5-fold'
                 ' (default: 1)',
            )
    parser.add_argument(
            '--settype', type=str, default='official-training',
            help='Which split to load: official-dev-[abs], official-training,'
                 ' internal-training, internal-dev'
                 ' (default: official-training)',
            )
    parser.add_argument(
            '--unit',    type=str, default='document',
            help='Unit to use with option --items_per_unit (`document` or `class`)'
                 ' (default: document)',
            )
    parser.add_argument(
            '--items_per_unit', type=int, default=1,
            help='How many instances (N) to create for each unit '
                 ' (deduplication may prevent reaching N; default: 1)',
            )
    parser.add_argument(
            '--fraction_of_subunits', type=float, default=0.0,
            help='What fraction of subunits, e.g. documents for each class, to include'
                 ' in each instance, 0.0 to 1.0; must be 0 or 1 for unit `document` as'
                 ' documents have no subunits (default: 0.0)',
            )
    parser.add_argument(
            '--number_of_subunits', type=int, default=0,
            help='What number of subunits, e.g. documents for each class, to include'
                 ' in each instance, 1 to ??; must be 0 for unit `document` as'
                 ' documents have no subunits (default: 0)',
            )
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser.add_argument('--deduplicate', action='store_true')
    parser.add_argument(
            '--no-deduplicate', dest='deduplicate', action='store_false',
            help='allow duplicate items to be created (default: deduplicate items)',
            )
    parser.set_defaults(deduplicate=True)
    args = parser.parse_args()
    if args.fraction_of_subunits == 0.0: args.fraction_of_subunits = None
    if args.number_of_subunits == 0.0: args.number_of_subunits = None
    data = EDOSDataset(
        args.seed,
        args.path, args.run, args.settype,
        unit = args.unit, items_per_unit = args.items_per_unit,
        fraction_of_subunits = args.fraction_of_subunits,
        number_of_subunits = args.number_of_subunits,
        deduplicate = args.deduplicate,
    )
    print('Number of items:', len(data))
    data.save_to_file(sys.stdout)

if __name__ == '__main__':
    main()
