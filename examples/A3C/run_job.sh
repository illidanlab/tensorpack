#!/bin/sh
if [ "$#" -ne 2 ]
then
    echo "./run_job.sh <env_name> <shaping>(0 for non-shaping, 1 for shallow, 2 for deep)"
    exit
else
    env=$1
    shaping=$2
    if [ "$shaping" = "0" ]; then
	    echo "python train-atari-from-scratch.py --env $env"
	    python train-atari-from-scratch.py --env $env
    elif [ "$shaping" = "1" ]; then
	    echo "python train-atari-reward-shaping.py --env $env --shaping shallow"
	    python train-atari-reward-shaping.py --env $env --shaping shallow
    else
	    echo "python train-atari-reward-shaping.py --env $env --shaping deep"
	    python train-atari-reward-shaping.py --env $env --shaping deep
    fi

fi
