
env="Gravitar-v0"
echo $env
## train a supervised agent using expert demonstration data
python train-atari-from-scratch.py --env $env
# train from scratch
# reward-shaping 1
#python train-atari-rs-exp6.py
#python train-atari-rs-exp7.py
