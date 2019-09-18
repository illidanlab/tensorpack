#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu

import argparse
import cv2
import gym
import multiprocessing as mp
import numpy as np
import os
import six
import sys
import uuid
import tensorflow as tf
from six.moves import queue

from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils.serialize import dumps
from tensorflow.python.tools import inspect_checkpoint as chkp

from atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState
from common import Evaluator, eval_model_multithread, play_n_episodes
from simulator import SimulatorMaster, SimulatorProcess, TransitionExperience


if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

FRAME_HISTORY = 4
IMAGE_SIZE = (84, 84)
STATE_SHAPE = IMAGE_SIZE + (3, )
GAMMA = 0.99

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 2000
EVAL_EPISODE = 10
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 16     # batch for efficient forward
SIMULATOR_PROC = 8#mp.cpu_count() * 2
PREDICTOR_THREAD_PER_GPU = 4
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None
model = None


def get_player(train=False, dumpdir=None):
    env = gym.make(ENV_NAME)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir, video_callable=lambda _: True)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    #env = RewardShaping(env)
    return env


class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)


class Model(ModelDesc):
    """
    Master model for training and predicting
    """
    def inputs(self):
        assert NUM_ACTIONS is not None
        return [tf.TensorSpec((None,) + STATE_SHAPE + (FRAME_HISTORY, ), tf.uint8, 'state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'futurereward'),
                tf.TensorSpec((None,), tf.float32, 'action_prob'),
                ]

    def _get_NN_prediction(self, state):
        assert state.shape.rank == 5  # Batch, H, W, Channel, History
        state = tf.transpose(state, [0, 1, 2, 4, 3])  # swap channel & history, to be compatible with old models
        image = tf.reshape(state, [-1] + list(STATE_SHAPE[:2]) + [STATE_SHAPE[2] * FRAME_HISTORY])

        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('master-conv0', image, 32, 5)
            l = MaxPooling('master-pool0', l, 2)
            l = Conv2D('master-conv1', l, 32, 5)
            l = MaxPooling('master-pool1', l, 2)
            l = Conv2D('master-conv2', l, 64, 4)
            l = MaxPooling('master-pool2', l, 2)
            l = Conv2D('master-conv3', l, 64, 3)

        l = FullyConnected('master-fc0', l, 512)
        l = PReLU('master-prelu', l)
        logits = FullyConnected('master-fc-pi', l, NUM_ACTIONS)    # unnormalized policy
        value = FullyConnected('master-fc-v', l, 1)
        return logits, value


    def build_graph(self, state, action, futurereward, action_prob):
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)

        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')
        value_beta = tf.get_variable('value_beta', shape=[],
                                       initializer=tf.constant_initializer(0.001), trainable=False)

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss * value_beta])
        #cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta])

        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        avg_futurereward = tf.reduce_mean(futurereward, name='avg_futurereward')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, 
                                   pred_reward, advantage,
                                   cost, avg_futurereward, tf.reduce_mean(importance, name='importance'))
        return cost


    def optimizer(self, scope_name="master"):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus, reward_shaping=False):
        """
        Args:
            gpus (list[int]): the gpus used to run inference
        """
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c, reward_shaping=reward_shaping)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

        self.reward_shaping=reward_shaping
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        state = tf.placeholder(dtype=tf.uint8, shape= (None,) + STATE_SHAPE + (FRAME_HISTORY, ) )
        self.rs_prob_dist = self._get_pretrained_module(state, STATE_SHAPE, FRAME_HISTORY)
        self.rs_state_ph = state
        ## TBD load parameter, or init parameter
        saver = tf.compat.v1.train.Saver()
        path = "/mnt/research/judy/reward_shaping/sanity/model_checkpoint"
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        logger.info("--" * 20 + "Model loaded successfully." + "--" * 20)


    def _get_pretrained_module(self, state, STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS=6):
        assert state.shape.rank == 5  # Batch, H, W, Channel, History
        state = tf.transpose(state, [0, 1, 2, 4, 3])  # swap channel & history, to be compatible with old models
        image = tf.reshape(state, [-1] + list(STATE_SHAPE[:2]) + [STATE_SHAPE[2] * FRAME_HISTORY])

        image = tf.cast(image, tf.float32) / 255.0


        with argscope(Conv2D, activation=tf.nn.relu):
            l = Conv2D('conv0', image, 32, 5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, 32, 5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, 64, 4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, 64, 3)

        l = FullyConnected('fc0', l, 512)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, NUM_ACTIONS)    # unnormalized policy
        logits = tf.nn.softmax(logits)
        logits = tf.stop_gradient(logits)
        return logits

    def _setup_graph(self):
        # Create predictors on the available predictor GPUs.
        num_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value'],
            self._gpus[k % num_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()
        logger.info("Starting MySimulatorMaster ...")
        start_proc_mask_signal(self)

    def _on_state(self, state, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            try:
                distrib, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client.memory.append(TransitionExperience(
                state, action, reward=None, real_reward=None, value=value, prob=distrib[action]))

            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, msg):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarded
        ####################
        # reward shaping
        ####################
        if not self.reward_shaping:
            client, state, reward, isOver = msg
            real_reward = reward
        else:
            client, state, action, reward, isOver = msg
            prob_dist = self.sess.run(
                [self.rs_prob_dist],
                feed_dict={
                    self.rs_state_ph : np.expand_dims(state, 0)
                }
            )
            logit = prob_dist[0][0][action]
            real_reward = reward
            reward += logit

        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            client.memory[-1].real_reward = real_reward

            if isOver:
                # should clear client's memory and put to queue
                self._parse_memory(0, client, True)
            else:
                if len(client.memory) == LOCAL_TIME_MAX + 1:
                    R = client.memory[-1].value
                    self._parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)



    def _parse_memory(self, init_r, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def get_training_dataflow(self):
        # the queue contains batched experience
        return BatchData(DataFromQueue(self.queue), BATCH_SIZE)


def train():
    assert tf.test.is_gpu_available(), "Training requires GPUs!"
    dirname = os.path.join('/mnt/research/judy/reward_shaping/sanity_reward_shaping/', 'train-atari-{}'.format(ENV_NAME))
    logger.set_logger_dir(dirname)

    #####################
    # assign GPUs for training & inference
    #num_gpu = get_num_gpu() - 1
    num_gpu = 1
    #####################
    global PREDICTOR_THREAD
    if num_gpu > 0:
        if num_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(num_gpu))[-num_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    #####################
    # setup actor process
    #####################
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]

    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)
    master = MySimulatorMaster(namec2s, names2c, predict_tower, reward_shaping=True)
    model = Model()
    config = TrainConfig(
        model=model,
        dataflow=master.get_training_dataflow(),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            master,
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_steps=2000),
        ],
        session_creator=sesscreate.NewSessionCreator(config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=SmartInit("/mnt/research/judy/reward_shaping/sanity/model_checkpoint/checkpoint"),
        max_epoch=1000,
    )
    trainer = SimpleTrainer() #if num_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model', default="/mnt/research/judy/reward_shaping/Pong-v0.npz", type=str)
    parser.add_argument('--env', help='env', default="Pong-v0", type=str)
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train', 'dump_video'], default='train')
    parser.add_argument('--output', help='output directory for submission', default='output_dir')
    parser.add_argument('--episode', help='number of episode to eval', default=1, type=int)
    parser.add_argument('--render', help='If render the environment', default=False, type=bool)
    parser.add_argument('--save', help='If save episodes', default=False, type=bool)
    parser.add_argument('--save_id', help='Index of Batches to be collected', default=1, type=int)
    parser.add_argument("--reward_shaping", help="If to use reward-shaping", default=True)
    args = parser.parse_args()

    ENV_NAME = args.env
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Environment: {}, number of actions: {}".format(ENV_NAME, NUM_ACTIONS))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=SmartInit(args.load),
            input_names=['state'],
            output_names=['policy']))
        if args.task == 'play':
            filename = "/mnt/research/judy/reward_shaping/expert_data/batch_{}.npz".format(args.save_id)
            play_n_episodes(get_player(train=False), pred,
                            args.episode, render=args.render, save=args.save, filename=filename)
        elif args.task == 'eval':
            eval_model_multithread(pred, args.episode, get_player)
        elif args.task == 'dump_video':
            play_n_episodes(
                get_player(train=False, dumpdir=args.output),
                pred, args.episode)
    else:
        train()
