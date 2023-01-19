#!/bin/bash

echo "The following two lines should match"
cut -d',' -f1 dev_task_a_labels.csv | md5sum
cut -d',' -f1 dev-task-a/dev_task_a_entries.csv | md5sum

echo "The following two lines should match"
cut -d',' -f1 dev_task_b_labels.csv | md5sum
cut -d',' -f1 dev-task-b/dev_task_b_entries.csv | md5sum

echo "The following two lines should match"
cut -d',' -f1 dev_task_c_labels.csv | md5sum
cut -d',' -f1 dev-task-c/dev_task_c_entries.csv | md5sum

echo "Extracting labels..."
cut -d',' -f2 dev_task_a_labels.csv > dev-task-a/labels.csv
cut -d',' -f2 dev_task_b_labels.csv > dev-task-b/labels.csv
cut -d',' -f2 dev_task_c_labels.csv > dev-task-c/labels.csv

echo "Combining labels..."
dev-task-a/
paste -d',' dev_task_a_entries.csv labels.csv > dev_task_a_combined.csv
cd ..
cd dev-task-b
paste -d',' dev_task_b_entries.csv labels.csv > dev_task_b_combined.csv
cd ..
cd dev-task-c/
paste -d',' dev_task_c_entries.csv labels.csv > dev_task_c_combined.csv
cd ..
cd ..
