Typical run:
``` shell
scripts/train.py
scripts/predict.py
./evaluate.py --newest data/k8020-dev-run-1.csv 
```

Voting ensemble:
``` shell
scripts/vote.py --weights 95,26,26,20,18     \
    predictions-opt-13b_dev_task_a-fixed.csv  \
    predictions-custom-hate-bert_dev_task_a-fixed.csv  \
    predictions-custom-hate-bert-punct-remove_dev_task_a-fixed.csv  \
    7314-xgboost-freq-12-dev-da7a53014f2ca82a.csv  \
    7254-linear-svm-c0025-freq-12-dev-6f7b5bdb6860f079.csv
./evaluate.py --newest data/dev-task-a/dev_task_a_combined.csv 
```
