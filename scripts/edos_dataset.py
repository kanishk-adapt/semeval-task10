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
        path = os.path.join(path, path_suffix)
        df = pandas.read_csv(path)
        # https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        for _, row in df.iterrows():
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
            #if self.debug: sys.stderr.write('Loaded document %s\n' %doc_id)
        self.is_labelled = is_labelled

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
