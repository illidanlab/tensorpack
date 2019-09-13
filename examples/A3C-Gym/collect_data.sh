#!/bin/bash
eps=10

shift=100
for i in 1 2 3 4 5 6 7 8 9 10
do
    sleep $i
    j=$(($i+$shift))
    ./train-atari.py --task play --env Pong-v0 --save True --save_id $j --episode $eps &

done; wait


