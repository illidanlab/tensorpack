#!/bin/bash
eps=10

shift=10
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=20
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=30
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=40
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=50
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=60
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=70
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=80
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait

shift=90
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    #./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &
    ./train-atari-gravitar.py --task play --env Gravitar-v0 --save True --save_id $j --episode $eps &

done; wait
