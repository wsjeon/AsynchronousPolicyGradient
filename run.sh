#!/bin/bash
# Asynchronous reinforcement learning
#
# Kill 'ps' job after training.
#
# Wonseok Jeon at KAIST
# wonsjeon@kaist.ac.kr

NUM_WORKERS=1

python main.py --job_name 'ps' --task_index 0 &
PID[0]=$!

task_index=0
while [ "$task_index" -lt "$NUM_WORKERS" ]; do
	python main.py --job_name 'worker' --task_index $task_index &
	PID[$(($task_index + 1))]=$!
	task_index=$(($task_index + 1))
done

task_index=0
while [ "$task_index" -lt "$NUM_WORKERS" ]; do
	task_index=$(($task_index + 1))
	wait ${PID[$task_index]}
done

kill ${PID[0]}
