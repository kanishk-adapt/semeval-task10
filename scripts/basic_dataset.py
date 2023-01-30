#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2019, 2022, 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on
# https://github.com/jowagner/mtb-tri-training/blob/master/scripts/basic_dataset.py

from __future__ import print_function

import bz2
import gzip
import collections
import hashlib
import os
import random
import sys
import time

from edos_labels import task_b_long_labels, task_c_short2long


short2long_tag_name = {
    't': 'token',
    'p': 'pos_tag',
    'd': 'dep_tag',
    's': 'sentiment',
}


class Item:

    def __init__(self, dataset, documents, label):
        self.dataset  = dataset
        self.documents = documents
        self.label    = label

    def get_documents(self):
        for doc_idx in self.documents:
            yield self.dataset.documents[doc_idx]

    def get_text_and_info(self, get_text = True, get_info = True):
        if not get_text and not get_info:
            return None
        text = []
        info = []
        for seq_idx, doc_idx in enumerate(self.documents):
            document = self.dataset.documents[doc_idx]
            if get_text: text.append(document)
            if get_info: info.append((doc_idx, len(document.split())))
        if get_text:
            text = ' '.join(text)
            if get_info:
                return text, info
            else:
                return text
        elif get_info:
            return info
        else:
            raise RuntimeError('get_text and/or get_info changed')

    def get_text(self):
        return self.get_text_and_info(get_info = False)

    def get_info(self):
        return self.get_text_and_info(get_text = False)

    def get_number_of_tokens(self):
        count = 0
        for _, n in self.get_text_and_info(get_text = False):
            count += n
        return count

    def get_tags(self, tag_combination):
        retval = []
        for doc_idx in self.documents:
            doc_tags = self.dataset.tags[doc_idx]
            retval += self.get_document_tags(doc_tags, tag_combination)
        return retval

    def get_document_tags(self, doc_tags, tag_combination):
        global short2long_tag_name
        exp_doc_tags = {}
        for tag_name in tag_combination:
            long_tag_name = short2long_tag_name[tag_name]
            exp_doc_tags[tag_name] = doc_tags[long_tag_name].split(' ')
        n_tags = len(exp_doc_tags[tag_combination[0]])
        retval = []
        for index in range(n_tags):
            comb_tag = []
            for tag_name in tag_combination:
                comb_tag.append(exp_doc_tags[tag_name][index])
            retval.append(' '.join(comb_tag))
        return retval

    def hexdigest(self, trim = None):
        if not trim:
            trim = 128
        h = hashlib.sha512()
        text = self.get_text()
        h.update(b'%x:' %len(text))  # length in characters
        h.update(text.encode('utf-8'))  # convert to binary
        return h.hexdigest()[:trim]

    def save_to_file(self, out, fileformat = 'debug', **kwargs):
        if fileformat == 'debug':
            self.save_to_file_for_debugging(out, **kwargs)
        elif fileformat == 'edos':
            self.save_to_file_in_edos_format(out, **kwargs)
        else:
            raise ValueError('unknown file format %s' %fileformat)

    def save_to_file_in_edos_format(self, out, index = None):
        # rewire_id,text,label_sexist,label_category,label_vector
        row = []
        # (1) rewire_id column
        if len(self.documents) == 1:
            doc_id = self.documents[0]
            row.append(self.dataset.docs[doc_id])
        else:
            if index is None:
                raise ValueError('When saving an augmented dataset in edos format, you need to provide the index of the item')
            row.append('concat-%d' %index)
        # (2) text column
        text = self.get_text()
        if '"' in text or ',' in text:
            # cell protection in csv format
            if '"' in text:
                text = text.replace('"', '""')
            text = '"%s"' %text
        row.append(text)
        # (3) label columns
        if self.label == 'none':
            row.append('not sexists')
            row.append('none')
            row.append('none')
        else:
            row.append('sexists')
            row.append(task_b_long_labels[int(self.label[0])-1])
            row.append(task_c_short2long[self.label])
        # (4) write row
        out.write(','.join(row))
        out.write('\n')

    def save_to_file_for_debugging(self, out):
        out.write('with %d document(s)' %len(self.documents))
        if self.label:
            out.write(' and label %s' %self.label)
        out.write('\n')
        text, info = self.get_text_and_info()
        header = 'index\tdoc_idx\tID\ttokens\n'
        out.write(header)
        seq_idx = 0
        for doc_idx, n_tokens in info:
            row = []
            row.append('%d' %seq_idx)
            row.append('%d' %doc_idx)
            doc_id = self.dataset.docs[doc_idx]
            row.append(doc_id)
            row.append('%d' %n_tokens)
            out.write('\t'.join(row))
            out.write('\n')
            seq_idx += 1
        wrapped_text, n_lines = self.dataset.wrap_text(text)
        out.write('text with %d lines follows\n' %n_lines)
        out.write(wrapped_text)
        out.write('\n')


class Dataset(collections.Sequence):

    """ Abstract base class for data sets.
    """

    def __init__(self, seed = None,
        path = None, run = 1, settype = 'test',
        skip_not_sexist = False,
        debug = False,
        **kwargs   # see set_mode()
    ):

        # (1) source data set:
        self.documents = []  # here we store the text
        self.doc2label = []  # for each document, record the label at the same
                             # index, i.e. maps doc_idx to human-readable label
                             # (may be replaced with an array)
        self.labels    = []  # list of human-readable labels
        self.docs      = []  # human-readable document IDs, i.e. maps
                             # doc_idx to document ID
        self.doc2idx   = {}  # map human-readable document IDs to doc_idx

        # (2) transformed data set:
        self.items       = []  # list of objects of class `Item`
        self.hash2item_idxs = {}  # speed up deduplication

        self.is_tokenised = 'unknown'

        if seed is None:
            seed = ''
        self.seed = seed.encode('utf-8')
        self.frozen = False
        self.debug  = debug
        self.set_mode(**kwargs)
        if path:
            self.load_from_path(path, run, settype, skip_not_sexist)
        self._reset_items()

    def freeze(self):
        self.frozen = True

    def set_mode(self,
        unit                 = 'document',
        items_per_unit       = 1,
        fraction_of_subunits = None,
        number_of_subunits   = None,
        deduplicate          = True,
    ):
        if self.frozen:
            raise ValueError('Cannot set dataset mode after freezing it')
        if items_per_unit > 100:
            sys.stderr.write('Warning: current implementation is not efficient for large number of samples')
        if fraction_of_subunits and number_of_subunits:
            raise ValueError('Cannot set both fraction_of_subunits and number_of_subunits')
        if fraction_of_subunits is not None and fraction_of_subunits == 0.0:
            raise ValueError('fraction_of_subunits must not be zero')
        if number_of_subunits is not None and number_of_subunits == 0.0:
            raise ValueError('number_of_subunits must not be zero')
        if unit == 'document':
            if items_per_unit != 1 and deduplicate:
                raise ValueError('Asking for %d copies of each document and deduplication' %items_per_unit)
            if (fraction_of_subunits is not None and (0.0 < fraction_of_subunits < 1.0)) \
            or number_of_subunits   is not None:
                if self.debug: sys.stderr.write('%r, %r\n' %(fraction_of_subunits, number_of_subunits))
                raise NotImplementedError  # currently no support for subunits of documents
        elif unit == 'class':
            if not fraction_of_subunits and not number_of_subunits:
                # we interpret this to mean "take all subunits"
                fraction_of_subunits = 1.0
            if items_per_unit != 1 and deduplicate and fraction_of_subunits == 1.0:
                raise ValueError('Asking for %d copies of all data per class and deduplication' %items_per_unit)
        else:
            raise ValueError('Unsupported unit %s' %unit)
        self.unit = unit
        self.items_per_unit = items_per_unit
        self.fraction_of_subunits = fraction_of_subunits
        self.number_of_subunits = number_of_subunits
        self.deduplicate = deduplicate
        self._reset_items()

    def _reset_items(self):
        if not self.documents:
            assert not self.items  # assumes we cannot delete documents
            return
        self.items = []
        if self.deduplicate:
            self.hash2item_idxs = {}
        if self.unit == 'class':
            self._reset_items_for_per_class_mode()
        elif self.unit == 'document':
            self._reset_items_for_per_document_mode()
        else:
            raise NotImplementedError

    def _add_item(self, item):
        if self.deduplicate:
            text = item.get_text()
            h = hash(text)  # Python's fast 64-bit hash function
            new_item_idx = len(self.items)
            if h in self.hash2item_idxs:
                for item_idx in self.hash2item_idxs[h]:
                    if text == self[item_idx].get_text():
                        # an item with the same text exists -->
                        # do not add this item to self.items
                        return False
                    #else: log.curiosity('found a hash collision')
            else:
                self.hash2item_idxs[h] = []
            self.hash2item_idxs[h].append(new_item_idx)
        self.items.append(item)
        return True

    def _reset_items_for_per_class_mode(self):
        for label in sorted(self.labels):
            # find candidate documents
            candidate_documents = []
            for doc_idx in range(len(self.documents)):
                if self.doc2label[doc_idx] == label:
                    candidate_documents.append(doc_idx)
            # create and expand samples
            for sample_of_documents in self._get_samples(
                candidate_documents,
                self.items_per_unit, self.fraction_of_subunits,
                self.number_of_subunits,
                extra_seed = label
            ):
                self._add_item(Item(self, sample_of_documents, label))

    def _reset_items_for_per_document_mode(self):
        assert self.fraction_of_subunits is None or self.fraction_of_subunits in (0.0, 1.0)
        assert self.number_of_subunits is None
        for doc_idx in range(len(self.documents)):
            label = self.doc2label[doc_idx] if self.is_labelled else None
            for copy in range(self.items_per_unit):
                self._add_item(Item(self, [doc_idx], label))

    def _get_samples(self, candidates, size, fraction_of_candidates, number_of_candidates, extra_seed = ''):
        extra_seed = extra_seed.encode('utf-8')
        total_candidates = len(candidates)
        debug = self.debug
        if debug and fraction_of_candidates:
            print('_get_samples() with %d candidates, size %s, fraction %.3f' %(total_candidates, size, fraction_of_candidates))
        if debug and number_of_candidates:
            print('_get_samples() with %d candidates, size %s, sample size %d' %(total_candidates, size, number_of_candidates))
        samples = []
        rounding_errors = [0.0, 0.0, 0.0, 0.0]
        for sample_index in range(size):
            if number_of_candidates:
                target_sample_size = number_of_candidates
                k_candidates       = number_of_candidates
            else:
                target_sample_size = total_candidates * fraction_of_candidates + rounding_errors[0]
                k_candidates = int(target_sample_size + 0.5)
            if k_candidates < 1:
                if debug: print('fraction_of_candidates too small with %d candidates, adjusting to 1' %total_candidates)
                k_candidates = 1
            if k_candidates > total_candidates:
                if debug: print('Adjusting k_candidates to %d (numerical rounding error or exceeding total)\n' %total_candidates)
                k_candidates = total_candidates
            del rounding_errors[0]
            error = target_sample_size - k_candidates
            rounding_errors.append(0.0)
            rounding_errors[0] += 7 * error / 16
            rounding_errors[1] += 5 * error / 16
            rounding_errors[2] += 3 * error / 16
            rounding_errors[3] += 1 * error / 16
            if debug: print('Sample [%d]: target %.3f k %d error %.3f diffusion %r' %(sample_index, target_sample_size, k_candidates, error, rounding_errors))
            # select k_candidates at random
            ranked_candidates = []
            for candidate_index, candidate in enumerate(candidates):
                h = hashlib.sha256()
                h.update(b'%d:' %len(self.seed))
                h.update(self.seed)
                h.update(b'%d:' %len(extra_seed))
                h.update(extra_seed)
                h.update(b'%d:' %total_candidates)
                h.update(b'%d:' %sample_index)
                h.update(b'%d:' %candidate_index)
                ranked_candidates.append((
                    h.hexdigest(),
                    candidate_index
                ))
            ranked_candidates.sort()
            sample = []
            for _, candidate_index in ranked_candidates[:k_candidates]:
                sample.append(candidates[candidate_index])
            samples.append(sample)
        return samples

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)

    def get_number_of_backing_documents(self):  # not all of these have to be in use in items
        return len(self.documents)

    def clone(self):
        ''' make new lists so that shuffling the clone
            does not affect self
        '''
        copy = Dataset(seed = self.seed.decode('utf-8'), path = None)
        copy.documents = self.documents
        copy.doc2label = self.doc2label
        copy.labels    = self.labels
        copy.docs      = self.docs
        copy.doc2idx   = self.doc2idx
        copy.items     = self.items[:]  # new list to support independent shuffling
        copy.is_tokenised = self.is_tokenised
        return copy

    def shuffle(self, rng):
        rng.shuffle(self.items)

    def load_from_path(self,
        path = 'data/name', run = 1, settype = 'test',
        skip_not_sexist = False,
        **kwargs
    ):
        """ Append data from `path` (usually a folder) to the data set.
            `kwargs` can be used to control subset selection via
            self.usable_document() defined in a derived class to which kwargs
            are passed.
        """
        raise NotImplementedError

    def open_for_reading(self, path):
        if os.path.exists(path + '.gz'):
            return gzip.open(path, 'rt', encoding = 'utf-8')
        if os.path.exists(path + '.bz2'):
            return bz2.open(path, 'rt', encoding = 'utf-8')
        return open(path, 'rt')

    def get_doc_idx(self, doc_id):
        try:
            doc_idx = self.doc2idx[doc_id]
        except KeyError:
            doc_idx = len(self.docs)
            self.docs.append(doc_id)
            self.doc2idx[doc_id] = doc_idx
        return doc_idx

    def usable_document(self, document, **kwargs):
        return True

    def wrap_text(self, text, width = 160):
        tokens = text.split()
        retval = []
        line   = []
        n_tokens = len(tokens)
        start = 0
        current_width = 0
        while start < n_tokens:
            token = tokens[start]
            if not line:
                line.append(token)
                current_width += len(token)
                start += 1
            elif current_width + 1 + len(token) <= width:
                line.append(token)
                current_width += (1 + len(token))
                start += 1
            else:
                retval.append(' '.join(line))
                line = []
                current_width = 0
        if line:
            retval.append(' '.join(line))
        n_lines = len(retval)
        return '\n'.join(retval), n_lines

    def save_to_file(self, out, item_filter = None, **kwargs):
        write_item_idx = False
        if 'fileformat' in kwargs and kwargs['fileformat'] == 'debug':
            write_item_idx = True
        if 'write_index' in kwargs:
            write_item_idx =  kwargs['write_index']
        for item_idx, item in enumerate(self):
            if item_filter and item_filter(item):
                continue  # skip item
            if write_item_idx:
                out.write('item %d\n' %item_idx)
            item.save_to_file(out, index = item_idx, **kwargs)

    def hexdigest(self, trim = None):
        if not trim:
            trim = 128
        h = hashlib.sha512()
        for item in self:
            text = item.get_text()
            h.update(b'%x:' %len(text))  # length in characters
            h.update(text.encode('utf-8'))  # convert to binary
        return h.hexdigest()[:trim]

    def get_number_of_tokens(self):
        count = 0
        for item in self:
            count += item.get_number_of_tokens()
        return count

    def get_text(self):
        retval = []
        for item in self:
            text = item.get_text()
            retval.append(text)
        return ' '.join(retval)

    def get_text_at_index(self, index):
        return self[index].get_text()


class ItemFilter:

    def __init__(self, target_columns,
        min_length = None, max_length = None,
        min_chars = None,  max_chars = None,
        skip_prob = 0.0, rng = None,
    ):
        self.min_length     = min_length
        self.max_length     = max_length
        self.min_chars      = min_chars
        self.max_chars      = max_chars
        self.skip_prob      = skip_prob
        self.rng            = rng

    def __call__(self, item):
        ''' returns True if the item should be skipped '''
        if self.skip_prob and self.rng.random() < self.skip_prob:
            return True
        text = item.get_text()
        num_items = len(text.split())
        if self.min_length and num_items < self.min_length:
            return True
        if self.max_length and num_items > self.max_length:
            return True
        if self.min_chars or self.max_chars:
            n_chars = len(text)
            if self.min_chars and n_chars < self.min_chars:
                return True
            if self.max_chars and n_chars > self.max_chars:
                return True
        return False


class Concat(Dataset):

    def __init__(self, datasets, item_modifier = None):
        self.datasets = datasets
        self.item_modifier = item_modifier
        self.items = []   # warning: different item format as in base class
        for ds_index, dataset in enumerate(datasets):
            if dataset is None:
                continue
            for d_index in range(len(dataset)):
                self.items.append((ds_index, d_index))

    def __getitem__(self, index):
        ds_index, d_index = self.items[index]
        item = self.datasets[ds_index][d_index]
        if self.item_modifier is not None:
            item = self.item_modifier(item)
        return item

    # len() of base class can be re-used as self.items is populated

    def clone(self):
        retval = Concat([])
        # make new lists so that shuffling the clone
        # does not affect self
        retval.items = self.items[:]
        retval.datasets = self.datasets[:]
        retval.item_modifier = self.item_modifier
        return retval

    def append(self, item):
        raise ValueError('Cannot append to concatenation')

    def load_from_path(self, *args):
        raise ValueError('Cannot load data into concatenation')


class DatasetSample(Dataset):

    def __init__(self, dataset, rng, size = None, percentage = None,
        with_replacement = True,
        unique_items = False,
        item_modifier = None,
        item_filter   = None,
        diversify_attempts = 1,
        disprefer = {},
        stratified = False,
        keep_order = False,
    ):
        # intentionally not calling init of base class as
        # we want to get an exception when members not
        # meaningful here are accessed
        if size and percentage:
            raise ValueError('Must not specify both size and percentage.')
        if percentage:
            size = int(0.5+percentage*len(dataset)/100.0)
        self.dataset = dataset
        self.frozen = True  # don't allow Dataset.set_mode()
        assert self.dataset.frozen
        self.is_vectorised = False
        self.item_modifier = item_modifier
        self.item_filter   = item_filter
        self.with_replacement  = with_replacement
        self.keep_order        = keep_order
        self.reset_sample(
            rng, size, diversify_attempts, disprefer,
            unique_items,
            stratified,
        )

    def _get_preferred_d_indices(self, d_size, size, disprefer, stratified = False):
        ''' size is the target size,
            d_size is the size of the existing dataset
        '''
        if size >= d_size or not disprefer:
            # use all data
            return list(range(d_size)), []
        # stratify data according to
        # how strongly items are dispreferred
        level2indices = {}
        max_level = 0
        for d_index in range(d_size):
            try:
                level = disprefer[d_index]
            except KeyError:
                level = 0
            if level not in level2indices:
                level2indices[level] = []
            level2indices[level].append(d_index)
            if level > max_level:
                max_level = level
        # select as much data as needed
        # starting with the lowest levels
        retval = []
        if stratified:
            # make sure the first and only iteration of the while loop
            # below when stratified is true adds at least 1 item to
            # retval
            level = min(level2indices.keys())
        else:
            level = 0
        # TODO: when filtering and/or rejecting duplicates, we may need
        #       more data but currently we are shuffling the data after
        #       calling this function, meaning that we must not return
        #       more data
        while len(retval) < size:
            assert level <= max_level, 'Missing some data after stratification.'
            try:
                indices = level2indices[level]
            except KeyError:
                indices = []
            retval += indices
            level += 1
            if stratified:
                break
        extra_data = []
        while level <= max_level:
            try:
                indices = level2indices[level]
            except KeyError:
                indices = []
            if stratified and indices:
                extra_data.append(indices)
            else:
                extra_data += indices
            level += 1
        if not stratified:
            new_extra_data = []
            new_extra_data.append(extra_data)
            extra_data = new_extra_data
        return retval, extra_data

    def reset_sample(
        self, rng, size = None,
        diversify_attempts = 1,
        disprefer = {},
        unique_items = False,
        stratified = False,
    ):
        if self.with_replacement and disprefer:
            # not clear how this should be implemented,
            # e.g. with what probability dispreferred
            # items should be picked
            raise NotImplementedError
        if self.keep_order and self.with_replacement:
            # this could be implemented by recording
            # the original index when sampling and
            # sorting the sample when finished
            raise NotImplementedError
        d_size = len(self.dataset)
        if size is None:
            size = d_size
            if not self.with_replacement:
                sys.stderr.write('Taking a sample of size 100% without replacement is just copying all data.\n')
        if unique_items and size > d_size:
            raise ValueError('Cannot make larger sample than given data without duplicating items')
        if not self.with_replacement:
            permutation, extra_data = self._get_preferred_d_indices(
                d_size, size, disprefer, stratified
            )
            p_size = len(permutation)
            if not self.keep_order:
                rng.shuffle(permutation)
        else:
            p_size = -1
            extra_data = []
        print('Sampling %s: %d target size, %d dataset size, %d permutation size, stratified is %r, %d dispreferred items, %d diversify_attempts, unique_items is %r, %d extra strata available' %(
            time.ctime(time.time()), size, d_size, p_size,
            stratified,
            len(disprefer), diversify_attempts, unique_items,
            len(extra_data)
        ), file=sys.stderr)
        self.items = []   # list of indices in self.dataset
        remaining = size
        rejected = 0
        filtered = 0
        previous_attempts_offset = 0
        if unique_items:
            so_far = set()
        last_verbose = time.time()
        interval = 5.0
        while remaining:
            candidates = []
            for attempt in range(diversify_attempts):
                if self.with_replacement:
                    # Since we currently do not support sampling with
                    # replacement together with disprefering some items,
                    # we can simply sample from all data:
                    d_index = rng.randrange(d_size)
                else:
                    p_index = size + previous_attempts_offset + \
                              attempt + filtered + \
                              rejected - remaining
                    if extra_data and p_index >= p_size:
                        e_data = extra_data[0]
                        del extra_data[0]
                        if not self.keep_order:
                            rng.shuffle(e_data)
                        permutation += e_data
                        old_p_size = p_size
                        p_size = len(permutation)
                        print('Sampling %s: extended %d permutation size to %d' %(
                            time.ctime(time.time()), old_p_size, p_size
                        ), file=sys.stderr)
                    d_index = permutation[p_index % p_size]

                if diversify_attempts == 1 or not self.items:
                    # no choice
                    priority = 0
                else:
                    priority = -self._nearest_neighbour_distance(d_index)
                candidates.append((priority, attempt, d_index))
            candidates.sort()
            _, attempt, d_index = candidates[0]
            previous_attempts_offset += attempt
            self.items.append(d_index)
            if unique_items or self.item_filter is not None:
                if time.time() > last_verbose + interval:
                    print('Sampling %s: %d left, %d rejected, %d filtered, %d target size, %d dataset size, %d permutation size' %(
                        time.ctime(time.time()), remaining, rejected, filtered,
                        size, d_size, p_size,
                    ), file=sys.stderr)
                    sys.stdout.flush()
                    last_verbose = time.time()
                    interval *= 2.0
            if unique_items \
            and p_index >= p_size * (1+diversify_attempts) \
            and not self.with_replacement:
                print('Sampling %s: giving up at p_index %d' %(
                    time.ctime(time.time()), p_index,
                ), file=sys.stderr)
                break
            dispr_offset = 10 + (d_index % 10)
            if self.item_filter is not None:
                if self.item_filter(self[-1]):
                    del self.items[-1]
                    filtered += 1
                    if disprefer:
                        # push item far down the list but not too far as
                        # _get_preferred_d_indices() iterates over the full
                        # range(min, max) of values
                        try:
                            disprefer[d_index] += dispr_offset
                        except KeyError:
                            disprefer[d_index] = dispr_offset
                    continue
            if unique_items:
                # check that the item just added is different from all so far:
                candidate_hash = self[-1].hexdigest(trim = 24)
                if candidate_hash in so_far:
                    del self.items[-1]
                    rejected += 1
                    if disprefer:
                        # push item far down the list but not too far as
                        # _get_preferred_d_indices() iterates over the full
                        # range(min, max) of values
                        try:
                            disprefer[d_index] += dispr_offset
                        except KeyError:
                            disprefer[d_index] = dispr_offset
                    continue
                # item is unique
                so_far.add(candidate_hash)
            remaining -= 1

    def _nearest_neighbour_distance(self, d_index):
        if not self.is_vectorised:
            self._vectorise()
        nn_distance = self._vector_distance(self[0], d_index)
        for candidate_item in self[1:]:
            distance = self._vector_distance(candidate_item, d_index)
            if distance < nn_distance:
                nn_distance = distance
        return nn_distance

    def _vectorise(self):
        self.vectors = []
        for item in self.dataset:
            self.vectors.append(item.get_vector_representation())
        self.is_vectorised = True

    def __getitem__(self, index):
        d_index = self.items[index]
        item = self.dataset[d_index]
        if self.item_modifier is not None:
            item = self.item_modifier(item)
        return item

    # len() of base class can be re-used as self.items is populated

    def indices(self):
        return self.items

    def clone(self, rnd = None):
        if rnd is None:
            rnd = random
        retval = DatasetSample([], rnd)
        # make new lists so that shuffling the clone
        # or changing the subset with reset_sample(),
        # set_counts() or set_remaining() does not
        # affect self
        retval.items = self.items[:]
        retval.dataset = self.dataset
        retval.is_vectorised = self.is_vectorised
        if self.is_vectorised:
            retval.vectors = self.vectors
        retval.item_modifier = self.item_modifier
        retval.with_replacement  = self.with_replacement
        return retval

    def append(self, item):
        raise ValueError('Cannot append to sample')

    def load_from_path(self, *args):
        raise ValueError('Cannot load data into sample')

    def get_counts(self):
        retval = len(self.dataset) * [0]
        for d_index in self.items:
            retval[d_index] += 1
        return retval

    def set_counts(self, rng, counts):
        self.items = []
        for d_index, count in enumerate(counts):
            for _ in count:
                self.items.append(d_index)
        self.shuffle(rng)

    def set_remaining(self, rng):
        '''
        Make this dataset the subset not selected by the current sample.
        '''
        counts = self.get_counts()
        self.items = []
        for d_index, count in enumerate(counts):
            if not count:
                self.items.append(d_index)
        self.shuffle(rng)


def new(**kwargs):
    ''' to be overwritten in corpus-specific dataset module to
        return a new Dataset object for the corpus initialised
        with `kwargs`
        '''
    raise NotImplementedError

def new_empty_set():
    raise NotImplementedError
    # if there was inheritance for modules we could write
    # `return new()`

def get_filter(**kwargs):
    ''' return an item filter object
        (must support that numerical values can be
        provided as strings)
    '''
    raise NotImplementedError
    # TODO: convert string to int
    #return ItemFilter([], **kwargs)

def test_error_diffusion(size = 1000):
    total_candidates = 100
    fraction_of_candidates = float(sys.argv[1])
    rounding_errors = [0.0, 0.0, 0.0, 0.0]
    for sample_index in range(size):
            target_sample_size = total_candidates * fraction_of_candidates + rounding_errors[0]
            k_candidates = int(target_sample_size + 0.5)
            del rounding_errors[0]
            error = target_sample_size - k_candidates
            rounding_errors.append(0.0)
            rounding_errors[0] += 7 * error / 16
            rounding_errors[1] += 5 * error / 16
            rounding_errors[2] += 3 * error / 16
            rounding_errors[3] += 1 * error / 16
            print(k_candidates)

if __name__ == '__main__':
    test_error_diffusion()
