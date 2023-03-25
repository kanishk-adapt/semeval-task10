<!-- ABOUT THE PROJECT -->
## About The REPO

This repoistory is a collaborative effort by the DCU - ADAPT Team for the
[EDOS 2023 SemEval Task-10](https://codalab.lisn.upsaclay.fr/competitions/7124)
shared task.

## Folder Structure

src/ -> scripts for all three tasks with transformer models
soc_sci_exp/ -> scripts and excel sheets with responses from 3 social science expert annotators and Chat-GPT and FLAN-T5 (large)
scripts -> scripts for experiments with baseline system and augmentation of training data

## Pre-requisites

Python3 modules:
gensim
nltk 
numpy
pandas
scikit-learn
spacy
xgboost
transformers
sentence-transformers
torch

(SpaCy is only needed if requesting its tokeniser with `train.py --tokeniser`,
or making predictions with a model that was trained with this option.)

NLTK asks to further run inside python:
```
import nltk
nltk.download('punkt')
```

Spacy instructions say to run
```
python -m spacy download en_core_web_sm
```
TODO: do we need this model? We only use the tokeniser.


## Usage

See task specific README.md in src/ directory


## License

See LICENSE.txt

## Citation

If you use this code or the released models, please cite our paper to appear at SemEval 2023.

## Acknowledgements
