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

from atari_wrapper import FireResetEnv, FrameStack, LimitLength, MapState 
from common import Evaluator, eval_model_multithread, play_n_episodes
from simulator import SimulatorMaster, SimulatorProcess, TransitionExperience

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
STATE_SHAPE = IMAGE_SIZE + (3, )

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
import settings


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

class DuelValueModel(ModelDesc):
    def inputs(self):
        assert NUM_ACTIONS is not None
        return [tf.TensorSpec((None,) + STATE_SHAPE + (FRAME_HISTORY, ), tf.uint8, 'state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'futurereward'),
                tf.TensorSpec((None,), tf.float32, 'futurereward_rs'),
                tf.TensorSpec((None,), tf.float32, 'action_prob'),
                ]

    def _get_NN_prediction(self, state):
        """
        Approximate two value functions: one for old MDP, one for shaped rewards MDP.
        """
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

            l2 = Conv2D('conv0', image, 32, 5)
            l2 = MaxPooling('pool0', l2, 2)
            l2 = Conv2D('conv1', l2, 32, 5)
            l2 = MaxPooling('pool1', l2, 2)
            l2 = Conv2D('conv2', l2, 64, 4)
            l2 = MaxPooling('pool2', l2, 2)
            l2 = Conv2D('conv3', l2, 64, 3)

        l0 = FullyConnected('master-fc0', l, 512)
        l0 = PReLU('master-prelu', l0)
        value = FullyConnected('master-fc-v', l0, 1) # value function for old MDP 
        logits = FullyConnected('master-fc-pi', l0, NUM_ACTIONS)    # unnormalized policy

        l1 = FullyConnected('master-fc0-rs', l, 512)
        l1 = PReLU('master-prelu-rs', l1)
        value_rs = FullyConnected('master-fc-v-rs', l1, 1) # value function for new MDP 


        l2 = FullyConnected('fc0', l2, 512)
        l2 = PReLU('prelu', l2)
        reward_shaping_logits = FullyConnected('fc-pi', l2, NUM_ACTIONS)    # unnormalized policy
        return logits, reward_shaping_logits, value, value_rs

    def build_graph(self, state, action, futurereward, futurereward_rs, action_prob):
        logits, reward_shaping_logits, value, value_rs = self._get_NN_prediction(state)
        reward_shaping_logits = tf.nn.softmax(reward_shaping_logits)
        reward_shaping_logits = tf.stop_gradient(reward_shaping_logits, name='reward_logits')
        
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        value_rs = tf.squeeze(value_rs, [1], name='pred_value_rs')  # (B,)

        policy = tf.nn.softmax(logits, name='policy')
        if not self.training:
            return

        log_probs = tf.log(policy + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        #advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')
        advantage = tf.subtract(tf.stop_gradient(value_rs), futurereward_rs, name='advantage-rs')

        avg_futurereward = tf.reduce_mean(futurereward, name='avg_futurereward')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1) 
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')

        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')
        value_loss_rs = tf.nn.l2_loss(value_rs - futurereward_rs, name='value_loss_rs')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        rms_advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss, value_loss_rs])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss, avg_futurereward,
                                   value_loss, value_loss_rs, pred_reward, rms_advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class DuelValueSimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus, args):
        """
        Args:
            gpus (list[int]): the gpus used to run inference
        """
        super(DuelValueSimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self.args = args
        self._gpus = gpus

    def _setup_graph(self):
        # Create predictors on the available predictor GPUs.
        num_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value', 'pred_value_rs', 'reward_logits'],
            self._gpus[k % num_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()
        logger.info("Starting DuelValueSimulatorMaster ...")
        start_proc_mask_signal(self)

    def _on_state(self, state, client):
        """
        Launch forward prediction for the new state given by some client.
        """
        def cb(outputs):
            try:
                distrib, value, value_rs, rs_distrib = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, value_rs=value_rs, prob=distrib[action], add_logit=rs_distrib[action]))
            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, msg):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarded
        client, state, action, reward, isOver = msg
        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            if isOver:
                # should clear client's memory and put to queue
                self.parse_memory(0, 0, client, True)
            else:
                if len(client.memory) == LOCAL_TIME_MAX + 1:
                    R = client.memory[-1].value
                    RS = client.memory[-1].value_rs
                    self.parse_memory(R, RS, client, False)
        # feed state and return action
        self._on_state(state, client)

    def parse_memory(self, init_r, initial_rs, client, isOver):
        if self.args.env.startswith("Pong"):
            self._parse_memory_with_cutoff(init_r, initial_rs, client, isOver)
        else:
            self._parse_memory(init_r, initial_rs, client, isOver)
    
    def _parse_memory_with_cutoff(self, init_r, initial_rs, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r) # old MPD value
        RS = float(initial_rs) # new MDP value
        for idx, k in enumerate(mem):
            logit2add = k.add_logit
            logit2add -= 1.0 / float(NUM_ACTIONS)

            if k.reward != 0: # we get one win/loss, stop counting future reward 
                R = np.clip(k.reward, -1, 1)
                RS2send = np.clip(k.reward + logit2add, -1, 1) 
            else: # it's not ending yet
                R = np.clip(np.clip(k.reward, -1, 1) + GAMMA * R, -1, 1)
                RS2send = np.clip(np.clip(RS + logit2add, -1, 1) + GAMMA * RS, -1, 1)

            self.queue.put([k.state, k.action, R, RS2send, k.prob])
            if 'deep' in self.args.shaping:
                RS = RS2send
            else:
                RS = R # only the current logit is added to discounted futurerewards

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def _parse_memory(self, init_r, initial_rs, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        RS = float(initial_rs)
        for idx, k in enumerate(mem):
            logit2add = k.add_logit
            logit2add -= 1.0 / float(NUM_ACTIONS)

            R = np.clip(np.clip(k.reward, -1, 1) + GAMMA * R, -1, 1)
            RS2send = np.clip(np.clip(k.reward + logit2add, -1, 1) + GAMMA * RS, -1, 1)
            self.queue.put([k.state, k.action, RS2send, k.prob])
            if self.args.shaping == 'deep':
                RS = RS2send
            else:
                RS = R
        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def get_training_dataflow(self):
        # the queue contains batched experience
        return BatchData(DataFromQueue(self.queue), BATCH_SIZE)

class Model(ModelDesc):
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

            l2 = Conv2D('conv0', image, 32, 5)
            l2 = MaxPooling('pool0', l2, 2)
            l2 = Conv2D('conv1', l2, 32, 5)
            l2 = MaxPooling('pool1', l2, 2)
            l2 = Conv2D('conv2', l2, 64, 4)
            l2 = MaxPooling('pool2', l2, 2)
            l2 = Conv2D('conv3', l2, 64, 3)

        l = FullyConnected('master-fc0', l, 512)
        l = PReLU('master-prelu', l)
        logits = FullyConnected('master-fc-pi', l, NUM_ACTIONS)    # unnormalized policy
        value = FullyConnected('master-fc-v', l, 1)

        l2 = FullyConnected('fc0', l2, 512)
        l2 = PReLU('prelu', l2)
        reward_shaping_logits = FullyConnected('fc-pi', l2, NUM_ACTIONS)    # unnormalized policy
        return logits, reward_shaping_logits, value

    def build_graph(self, state, action, futurereward, action_prob):
        logits, reward_shaping_logits, value = self._get_NN_prediction(state)
        reward_shaping_logits = tf.nn.softmax(reward_shaping_logits)
        reward_shaping_logits = tf.stop_gradient(reward_shaping_logits, name='reward_logits')
        
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        if not self.training:
            return

        log_probs = tf.log(policy + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')
        avg_futurereward = tf.reduce_mean(futurereward, name='avg_futurereward')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1) 
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = tf.sqrt(tf.reduce_mean(tf.square(advantage)), name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        cost = tf.truediv(cost, tf.cast(tf.shape(futurereward)[0], tf.float32), name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss, avg_futurereward,
                                   value_loss, pred_reward, advantage,
                                   cost, tf.reduce_mean(importance, name='importance'))
        return cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_norm(grad, 0.1 * tf.cast(tf.size(grad), tf.float32))),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, gpus, args):
        """
        Args:
            gpus (list[int]): the gpus used to run inference
        """
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self.args = args
        self._gpus = gpus

    def _setup_graph(self):
        # Create predictors on the available predictor GPUs.
        num_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value', 'reward_logits'],
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
                distrib, value, rs_distrib = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(client.ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action], add_logit=rs_distrib[action]))
                #state, action, reward=None, real_reward=None, value=value, prob=distrib[action]))
            self.send_queue.put([client.ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _process_msg(self, msg):
        """
        Process a message sent from some client.
        """
        # in the first message, only state is valid,
        # reward&isOver should be discarded
        client, state, action, reward, isOver = msg
        if len(client.memory) > 0:
            client.memory[-1].reward = reward
            if isOver:
                # should clear client's memory and put to queue
                self.parse_memory(0, client, True)
            else:
                if len(client.memory) == LOCAL_TIME_MAX + 1:
                    R = client.memory[-1].value
                    self.parse_memory(R, client, False)
        # feed state and return action
        self._on_state(state, client)

    def parse_memory(self, init_r, client, isOver):
        if self.args.env.startswith("Pong"):
            self._parse_memory_with_cutoff(init_r, client, isOver)
        else:
            self._parse_memory(init_r, client, isOver)
    
    def _parse_memory_with_cutoff(self, init_r, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            logit2add = k.add_logit
            logit2add -= 1.0 / float(NUM_ACTIONS)

            if k.reward != 0: # we get one win/loss, stop counting future reward 
                R = np.clip(k.reward, -1, 1)
            else: # it's not ending yet
                R = np.clip(k.reward, -1, 1) + GAMMA * R
            R2send = np.clip(R + logit2add, -1, 1)
            self.queue.put([k.state, k.action, R2send, k.prob])
            if self.args.shaping == 'deep':
                R = R2send

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def _parse_memory(self, init_r, client, isOver):
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            logit2add = k.add_logit
            logit2add -= 1.0 / float(NUM_ACTIONS)

            R = np.clip(k.reward, -1, 1) + GAMMA * R
            R2send = np.clip(R + logit2add, -1, 1)
            self.queue.put([k.state, k.action, R2send, k.prob])
            if self.args.shaping == 'deep':
                R = R2send
        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def get_training_dataflow(self):
        # the queue contains batched experience
        return BatchData(DataFromQueue(self.queue), BATCH_SIZE)


def train_duel_value(args):
    logger.info("Test")
    assert tf.test.is_gpu_available(), "Training requires GPUs!"
    if not args.logit_render_model_checkpoint:
        #args.logit_render_model_checkpoint = os.path.join(settings.supervised_model_checkpoint[args.env], 'checkpoint') 
        args.logit_render_model_checkpoint = settings.pretraind_model_path[args.env] 
    dirname = os.path.join(settings.path_prefix, "reward_shaping_model/env-{}-shaping-{}")
    dirname = dirname.format(args.env, args.shaping) 
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    num_gpu = args.num_gpu 
    global PREDICTOR_THREAD
    if num_gpu > 1:
        # use half gpus for inference
        predict_tower = list(range(num_gpu))[-num_gpu // 2:]
    else:
        predict_tower = [0]
    PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
    train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
    logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
        ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = DuelValueSimulatorMaster(namec2s, names2c, predict_tower, args)
    config = TrainConfig(
        model=DuelValueModel(),
        dataflow=master.get_training_dataflow(),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            master,
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                #EVAL_EPISODE, ['state'], ['reward_logits'], get_player),
                every_k_steps=2000),
        ],
        session_creator=sesscreate.NewSessionCreator(config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=SmartInit(args.logit_render_model_checkpoint),
        max_epoch=1000,
    )
    trainer = SimpleTrainer() #if num_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)

def train(args):
    assert tf.test.is_gpu_available(), "Training requires GPUs!"
    if not args.logit_render_model_checkpoint:
        #args.logit_render_model_checkpoint = os.path.join(settings.supervised_model_checkpoint[args.env], 'checkpoint') 
        args.logit_render_model_checkpoint = settings.pretraind_model_path[args.env] 
    dirname = os.path.join(settings.path_prefix, "reward_shaping_model/env_{}_shaping_{}")
    dirname = dirname.format(args.env, args.shaping) 
    logger.set_logger_dir(dirname)

    # assign GPUs for training & inference
    num_gpu = args.num_gpu 
    global PREDICTOR_THREAD
    if num_gpu > 1:
        # use half gpus for inference
        predict_tower = list(range(num_gpu))[-num_gpu // 2:]
    else:
        predict_tower = [0]
    PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
    train_tower = list(range(num_gpu))[:-num_gpu // 2] or [0]
    logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
        ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    prefix = '@' if sys.platform.startswith('linux') else ''
    namec2s = 'ipc://{}sim-c2s-{}'.format(prefix, name_base)
    names2c = 'ipc://{}sim-s2c-{}'.format(prefix, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, predict_tower, args)
    config = TrainConfig(
        model=Model(),
        dataflow=master.get_training_dataflow(),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            master,
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                #EVAL_EPISODE, ['state'], ['reward_logits'], get_player),
                every_k_steps=2000),
        ],
        session_creator=sesscreate.NewSessionCreator(config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        session_init=SmartInit(args.logit_render_model_checkpoint),
        max_epoch=1000,
    )
    trainer = SimpleTrainer() #if num_gpu == 1 else AsyncMultiGPUTrainer(train_tower)
    launch_train_with_config(config, trainer)


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='env', default="Pong-v0", type=str)
    parser.add_argument('--num_gpu', help='Number of GPUs', default=1, type=int)
    parser.add_argument('--logit_render_model_checkpoint', help='logit_render_model_checkpoint', default=None, type=str)
    parser.add_argument('--shaping', help='shaping approach',
                        choices=['shallow', 'deep', 'duel-shallow', 'duel-deep'], default='shallow', type=str)
    parser.add_argument('--task', help='task to perform',
                        choices=['eval', 'train'], default='train')
    parser.add_argument('--expert_data_id', help='file id to save expert data', default=1, type=int)
    parser.add_argument('--expert_episode', help='number of expert episodes to eval', default=1, type=int)
    parser.add_argument('--render', help='If render the environment', default=False, type=bool)
    parser.add_argument('--expert_save', help='If save episodes', default=False, type=bool)
    parser.add_argument('--expert_model_path', help='Model used to evaluate expert demonstration', default=None)

    args = parser.parse_args()

    ENV_NAME = args.env
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Environment: {}, number of actions: {}".format(ENV_NAME, NUM_ACTIONS))

    if args.task != 'train':
        pass
        #assert args.load is not None
        #pred = OfflinePredictor(PredictConfig(
        #    model=Model(),
        #    session_init=SmartInit(args.load),
        #    input_names=['state'],
        #    output_names=['policy']))
        #if args.task == 'play':
        #    filename = "/mnt/research/judy/reward_shaping/expert_data/batch_{}.npz".format(args.save_id)
        #    play_n_episodes(get_player(train=False), pred,
        #                    args.episode, render=args.render, save=args.save, filename=filename)
        #elif args.task == 'eval':
        #    eval_model_multithread(pred, args.episode, get_player)
        #elif args.task == 'dump_video':
        #    play_n_episodes(
        #        get_player(train=False, dumpdir=args.output),
        #        pred, args.episode)
    elif 'duel' in args.shaping:
        train_duel_value(args)
    else:
        train(args)
