Typical run: 

``` shell

python3 main.py --train_path k8020-tr.csv \
	--test_path task_c_merged.csv \
	--task_test_path test_task_c_entries.csv \
	--algorithm ['lr', 'knn'] \
	--model_dir <model_directory> \
	--run_number [1, 101, 102, 103, 104, 105]

```
