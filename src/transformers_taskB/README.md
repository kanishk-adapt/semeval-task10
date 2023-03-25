Typical run to TRAIN:
``` shell

python3 main_save.py --train_path k8020-tr.csv \
        --validation_path k8020-dev.csv \
        --test_path task_a_labels.csv \
	--label_number [1, 2, 3, 4] \
        --model_name ['bert', 'hate-bert', 'opt 350', 'opt 1.3', 'distil bert'] \
        --model_dir ['bert-base-uncased', '<path-to-hate-bert>', 'facebook/opt-350m', 'facebook/opt-1.3b', 'distilbert-base-uncased', ] \
        --batch_size <batch-size> \
        --epochs <epochs> \
        --learning_rate <learning-rate> \
        --weight_decay <weight-decay> \
        --wandb_project <wandb-project-name> \
        --save_dir <directory-path>

```
Typical run to TEST:

``` shell


python3 test.py --test_path test_task_b_entries.csv \
	--model_name ['opt', 'hbert'] \
	--model_dir_lab1 <directory_saved_model_label_1> \
	--model_dir_lab2 <directory_saved_model_label_2> \
	--model_dir_lab3 <directory_saved_model_label_3> \
	--model_dir_lab4 <directory_saved_model_label_4>
```


