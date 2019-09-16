#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu

import multiprocessing as mp
import os
import threading
import tensorflow as tf
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import six
import zmq
from six.moves import queue
from tensorpack import *
from tensorpack.utils import logger
from tensorpack.utils.concurrency import LoopThread, enable_death_signal, ensure_proc_terminate
from tensorpack.utils.serialize import dumps, loads
import numpy as np

__all__ = ['SimulatorProcess', 'SimulatorMaster', 'RewardShapingSimulator',
           'TransitionExperience']
GAMMA = 0.99
dirname = '/mnt/research/judy/reward_shaping/sanity/'

class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)



@six.add_metaclass(ABCMeta)
class SimulatorProcess(mp.Process):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        Args:
            idx: idx of this process
            pipe_c2s, pipe_s2c (str): name of the pipe
        """
        super(SimulatorProcess, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        enable_death_signal()
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        s2c_socket.connect(self.s2c)

        state = player.reset()
        reward, action, isOver = 0, 1, False
        while True:
            # after taking the last action, get to this state and get this reward/isOver.
            # If isOver, get to the next-episode state immediately.
            # This tuple is not the same as the one put into the memory buffer
            c2s_socket.send(dumps(
                (self.identity, state, action, reward, isOver)),
                copy=False)
            action = loads(s2c_socket.recv(copy=False))
            state, reward, isOver, _ = player.step(action)
            if isOver:
                state = player.reset()

    @abstractmethod
    def _build_player(self):
        pass



#@six.add_metaclass(ABCMeta)
class RewardShapingSimulator(threading.Thread):
    """ TBD """

    def build_graph(self, STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS):

        ## create graph, session
        tf.reset_default_graph()
        self.sess = tf.Session()
        state = tf.placeholder(dtype=tf.uint8, shape= (None,) + STATE_SHAPE + (FRAME_HISTORY, ) )

        logits = self._build_model(state, STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS)
        policy = tf.nn.softmax(logits, name='policy')
        ## calcualte the logit get by this action

        ## TBD load parameter, or init parameter
        saver = tf.compat.v1.train.Saver()
        sl_model_dirname = os.path.join(dirname, "model_checkpoint")
        saver.restore(self.sess, tf.train.latest_checkpoint(sl_model_dirname))
        print('logit provider model loaded and built successfully')
        results = {}
        results["policy"] = policy
        results["logits"] = logits
        results["states_ph"] = state
        results["saver"] = saver
        self.handler = results

    def _build_model(self, state, STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS):
        """
        state is a placeholder
        """
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
        return logits 

    def __init__(self, STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS, queue_m2r, queue_r2m):
        """
        Args:
            pipe_c2s, pipe_s2c (str): names of pipe to be used for communication
        """
        super(RewardShapingSimulator, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.daemon = True
        self.name = 'RewardShapingSimulator'
        self.build_graph(STATE_SHAPE, FRAME_HISTORY, NUM_ACTIONS)

        self.queue_m2r = queue_m2r
        self.queue_r2m = queue_r2m

    def run(self):
        try:
            while True:
                ## get (s,a ) from master
                #state, action = loads(self.m2r_socket.recv(copy=False))
                state, action = self.queue_m2r.get(block=True, timeout=None)
                ## evalate logit
                logit = self.predict_logits(state, action)
                #logger.info("Logit is: {}".format(logit))
                ## send logit back
                self.queue_r2m.put(logit)
                #self.r2m_socket.send(dumps((logit,)), copy=False)

        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    def predict_logits(self, state, action):
        state = np.expand_dims(state, 0)  # batch
        policy_vec = self.sess.run(
            [
                self.handler["policy"],
            ], 
            feed_dict={
                self.handler["states_ph"]:state,
                }
                )
        logit = policy_vec[0][0][action] 
        return logit


@six.add_metaclass(ABCMeta)
class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all SimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = []    # list of Experience
            self.ident = None

    def __init__(self, pipe_c2s, pipe_s2c, reward_shaping=False, queue_m2r=None, queue_r2m=None):
        """
        Args:
            pipe_c2s, pipe_s2c (str): names of pipe to be used for communication
        """
        super(SimulatorMaster, self).__init__()
        assert os.name != 'nt', "Doesn't support windows!"
        self.reward_shaping = reward_shaping
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        if reward_shaping:
            self.queue_m2r = queue_m2r
            self.queue_r2m = queue_r2m

        ## if reward_shaping:
        ##     if not (pipe_r2m and pipe_r2m):
        ##         logger.info("Pipes for reward-shaping must be set.")
        ##         exit(1)
        ##     self.rs_context = zmq.Context()
        ##     self.r2m_socket = self.rs_context.socket(zmq.PULL)
        ##     self.r2m_socket.bind(pipe_r2m)
        ##     self.r2m_socket.set_hwm(2)
        ##     self.m2r_socket = self.rs_context.socket(zmq.PUSH)
        ##     self.m2r_socket.bind(pipe_m2r)
        ##     self.m2r_socket.set_hwm(2)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)
        #if reward_shaping:
        #    atexit.register(clean_context, [self.r2m_socket, self.m2r_socket], self.rs_context)


    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False))
                ident, state, action, reward, isOver = msg
                client = self.clients[ident]
                if client.ident is None:
                    client.ident = ident
                # maybe check history and warn about dead client?
                if self.reward_shaping:
                    self.queue_m2r.put((state, action))
                    logit = self.queue_r2m.get()
                    #self.m2r_socket.send(dumps((state, action)), copy=False)
                    #logit = loads(self.r2m_socket.recv(copy=False))
                    #logit = 10
                    reward += logit
                self._process_msg(client, state, reward, isOver)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _process_msg(self, client, state, reward, isOver):
        pass

    def __del__(self):
        self.context.destroy(linger=0)


if __name__ == '__main__':
    import random
    import gym

    class NaiveSimulator(SimulatorProcess):
        def _build_player(self):
            return gym.make('Breakout-v0')

    class NaiveActioner(SimulatorMaster):
        def _get_action(self, state):
            time.sleep(1)
            return random.randint(1, 3)

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0

    name = 'ipc://@whatever'
    procs = [NaiveSimulator(k, name) for k in range(10)]
    [k.start() for k in procs]

    th = NaiveActioner(name)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
