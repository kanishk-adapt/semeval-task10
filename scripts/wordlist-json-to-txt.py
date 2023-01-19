#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import json
import sys

def extract_clusters(json_path, name):
    f = open(json_path, 'rt')
    data = json.load(f)
    f.close()
    for key in data:
        fields = key.split()
        assert len(fields) == 2
        assert fields[0] == 'Cluster'
        cluster_number = int(fields[1])
        out = open('%s-%d.txt' %(name, cluster_number), 'wt')
        for word in sorted(data[key]):
            out.write(word + '\n')
        out.close()

def main():
    extract_clusters('5_hbert_unlab_clusters.json',
                     'hbert-neg')
    extract_clusters('5_hbert_unlab_clusters_positive.json',
                     'hbert-pos')
    extract_clusters('5_opt_13b_clusters.json',
                     'opt13b-neg')
    extract_clusters('5_opt_13b_clusters_positive.json',
                     'opt13b-pos')

if __name__ == '__main__':
    main()

