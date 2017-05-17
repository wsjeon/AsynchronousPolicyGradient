#!/bin/bash
# Asynchronous learning using shell script.
#
# Kill 'ps' job after training. 
#
# Wonseok Jeon at KAIST
# wonsjeon@kaist.ac.kr
python main.py --job_name 'ps' --task_index 0 &
PID=$!
python main.py --job_name 'worker' --task_index 0
kill $PID
