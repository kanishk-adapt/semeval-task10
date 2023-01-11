#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import sys

def simple_tokeniser(text):
    return text.split()

# TODO: add a tokeniser that splits off punctuation and symbols
#       and maybe also a BERT-like tokeniser
