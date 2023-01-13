#!/usr/bin/env python
# -*- coding: utf-8 -*-

# (C) 2023 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
import sys

def simple_tokeniser(text):
    return text.split()

nltk_tokeniser = word_tokenize

def gensim_tokeniser(text):
    return list(tokenize(text))

spacy_nlp = None

def spacy_tokeniser(text):
    global spacy_nlp
    if spacy_nlp is None:
        from spacy.lang.en import English
        spacy_nlp = English()
    return [x.text for x in spacy_nlp(text)]

# TODO: add a BERT-like tokeniser
