#!/bin/bash
for job_name in 'ps' 'worker'
do
for task_index in 0
do
	python main.py --job_name $job_name --task_index $task_index &
done
done
