#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu

import argparse
import cv2
import gym
import multiprocessing as mp
import numpy as np
import pickle
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
import random 


if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

import settings
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
STATE_SHAPE = IMAGE_SIZE + (3, )
BATCH_SIZE = 128

def process_rewards(env_name, rewards):
    episodes = len(rewards) # how many episodes are in this file
    if env_name.startswith("Pong"):
        discounted_rewards = [] 
        for i in range(episodes):
            rs = rewards[i]
            discounted_r = np.zeros((rs.shape[0], 1))
            rs = np.clip(rs, -1, 1) 
            R = 0
            for t in reversed(range(len(rs))):
                if rs[t] == 0:
                    R = R * GAMMA + rs[t]
                else:
                    R = rs[t]
                discounted_r[t] = R
            discounted_rewards.append(discounted_r)
        return discounted_rewards
    else:
        discounted_rewards = [] 
        for i in range(episodes):
            rs = rewards[i]
            discounted_r = np.zeros((rs.shape[0], 1))
            rs = np.clip(rs, -1, 1) 
            R = 0
            for t in reversed(range(len(rs))):
                R = R * GAMMA + rs[t]
                discounted_r[t] = R
            discounted_rewards.append(discounted_r)
        return discounted_rewards


def find_available_data(args):
    # filename = os.path.join(settings.expert_data_path, "batch_{}.npz").format(args.expert_data_id)
    pathdir = settings.expert_data_path[args.env]
    file_ids = []
    for file in os.listdir(pathdir):
        if file.endswith(".npz"):
            file_id = file.strip(".npz").split("_")[1] 
            file_ids.append(file_id)
    return file_ids

def get_player(train=False, dumpdir=None):
    env = gym.make(ENV_NAME)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir, video_callable=lambda _: True)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    return env


class SupervisedModel(ModelDesc):
    def inputs(self):
        assert NUM_ACTIONS is not None
        return [tf.TensorSpec((None,) + STATE_SHAPE + (FRAME_HISTORY, ), tf.uint8, 'state'),
                tf.TensorSpec((None,), tf.int64, 'action'),
                tf.TensorSpec((None,), tf.float32, 'futurereward'),
                ]


    def _get_NN_prediction(self, state):
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

    def build_graph(self, resume=False):
        ## create graph, session
        tf.reset_default_graph()
        sess = tf.Session()
        action = tf.placeholder(dtype=tf.int64, shape=(None,1))
        state = tf.placeholder(dtype=tf.uint8, shape= (None,) + STATE_SHAPE + (FRAME_HISTORY, ) )
        futurereward = tf.placeholder(dtype=tf.float32, shape=(None,1))

        logits = self._get_NN_prediction(state)
        policy = tf.nn.softmax(logits, name='policy')
        log_probs = tf.log(policy + 1e-6)
        one_hot_actions = tf.one_hot(action, NUM_ACTIONS)
        one_hot_actions = tf.reshape(one_hot_actions, [-1, NUM_ACTIONS])
        policy_loss = tf.losses.softmax_cross_entropy(
            one_hot_actions, # one-hot-labels
            logits, # logits
        )
        confience_a_given_s = tf.reduce_mean(
            tf.reduce_sum(
            policy * one_hot_actions, 1)
        )
        cost = policy_loss 

        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)

        optimizer_op = tf.train.AdamOptimizer(lr, epsilon=1e-3).minimize(cost)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", cost)
        # Create a summary to monitor confidence tensor
        tf.summary.scalar("mean_pi_a_given_s", confience_a_given_s)
        # Merge all summaries into a single op
        merged = tf.summary.merge_all()

        ## load parameter, or init parameter
        saver = tf.compat.v1.train.Saver()
        if resume:
            print('loading and building pretrained policy')
            #saver.restore(sess, tf.train.latest_checkpoint(settings.supervised_model_checkpoint[self.args.env]))
            checkpoint = settings.supervised_model_checkpoint[self.args.env]
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            print('loaded and built successfully')
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print('model initialized successfully')


        writer = tf.compat.v1.summary.FileWriter(self.args.supervised_model_path, sess.graph)

        results = {}
        results["cost"] = cost
        results["policy"] = policy
        results["logits"] = logits
        results["merged"] = merged
        results["writer"] = writer

        results["actions_ph"] = action
        results["futurereward_ph"] = futurereward
        results["states_ph"] = state

        results["optimizer"] = optimizer_op
        results["saver"] = saver
        self.handler = results
        self.sess = sess

    def train(self, file_ids, epoches=1, initial_episode=0):
        episode_index = initial_episode
        for epoch in range(epoches):
            for file_id in file_ids:
                try:
                    states, actions, rewards = self.load_data(file_id=file_id)
                except AttributeError:
                    logger.info("Skipping file {}".format(file_id))
                    continue
                rewards = process_rewards(self.args.env, rewards) # get discounted rewards
                ## start training 

                for e in range(episodes):
                    episode_index += 1
                    # get each episode
                    print("File id = {}, Episode id ={}".format(file_id, episode_index))
                    e_state, e_action, e_reward = states[e], actions[e], rewards[e]
                    # state steps should be 1 more than action/reward steps
                    stride = BATCH_SIZE  
                    pos, frame_size = 0, len(e_action)
                    while True:                                                           
                        end = frame_size if pos+stride>=frame_size else pos+stride        
                        batch_x = np.reshape(e_state[pos:end], (-1,)  + STATE_SHAPE + (FRAME_HISTORY,)   )                                       
                        batch_y = np.reshape(e_action[pos:end], (-1, 1))                                            
                        batch_r = np.reshape(e_reward[pos:end], (-1,1)) 
                        _, loss_val, tf_summary = self.sess.run(
                            [
                                self.handler["optimizer"], 
                                self.handler["cost"],
                                self.handler["merged"],
                            ], 
                            feed_dict={
                                self.handler["states_ph"]:batch_x,
                                self.handler["futurereward_ph"]:batch_r,
                                self.handler["actions_ph"]:batch_y
                                }
                                )
                        pos = end
                        ## release memory space for each mini-batch
                        del batch_x, batch_y, batch_r
                        if pos >= frame_size:
                            # end of pisode
                            break 
                        ## print("Weight value: ", weight)
                    information = "Update Episode {:2d}, Episode Length {:5d}, Running Loss {:.4f}".format(episode_index, frame_size, loss_val) 
                    logger.info(information)
                    self.handler["writer"].add_summary(tf_summary, episode_index)

                    ## save session and Episode index
                    self.handler["saver"].save(self.sess, os.path.join(settings.supervised_model_checkpoint[self.args.env], "checkpoint.ckpt") )
                    fp = open(os.path.join(settings.supervised_model_checkpoint[self.args.env], "step.p"), "wb")
                    pickle.dump(episode_index, fp)
                    fp.close()

                del states, actions, rewards


    def load_data(self, file_id):
        pathdir = settings.expert_data_path[self.args.env] 
        path = os.path.join(pathdir, "batch_{}.npz".format(file_id))
        data = np.load(path, allow_pickle=True)
        states = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        return states, actions, rewards

    def evaluate(self):
        self.env = get_player()
        for episode in range(args.expert_episode):
            score = self._evaluate_one_episode()
            logger.info("Episode {} Player Score: {:.1f}".format(episode, score))
    
    def _evaluate_one_episode(self):
        ob = self.env.reset()
        isOver = False
        sum_r = 0
        while not isOver:
            # get prediction
            ob = np.expand_dims(ob, 0)  # batch
            policy = self.sess.run(
                [
                    self.handler["policy"],
                ], 
                feed_dict={
                    self.handler["states_ph"]:ob,
                    }
                    )
            ## get actions based on prediction
            act = np.argmax(policy)
            if random.random() < 0.01: # eplison-greedy
                spc = self.env.action_space
                act = spc.sample()

            ## step with the environment
            ob, r, isOver, info = self.env.step(act)
            if self.args.render:
                self.env.render()
            sum_r += r
            if isOver:
                return sum_r





def train(args):
    assert tf.test.is_gpu_available(), "Training requires GPUs!"
    logger.set_logger_dir(args.supervised_model_path)

    # assign GPUs for training & inference
    num_gpu = args.num_gpu
    # setup model 
    model=SupervisedModel()
    model.args = args
    model.build_graph(resume=args.resume)
    # training model using loaded expert data
    file_ids = find_available_data(args)
    step_file = os.path.join(settings.supervised_model_checkpoint[args.env], "step.p")
    if args.resume and os.path.exists(step_file): 
        with open(step_file, 'rb') as f:
            initial_episode = pickle.load(f)
    else:
        initial_episode = 0
    model.train(file_ids, epoches=args.train_epochs, initial_episode=initial_episode)


def generate_expert_demonstration(args):
    logger.info("Loaded Model Path: {}".format(args.expert_model_path))
    Model = SupervisedModel()
    Model.args = args
    Model.build_graph(resume=True)
    Model.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='env', default="Pong-v0", type=str)
    parser.add_argument('--num_gpu', help='Number of GPUs', default=1, type=int)
    parser.add_argument('--task', help='task to perform',
                        choices=['eval', 'train'], default='train')
    parser.add_argument('--resume', help='If Resume Training Supervised Model', default=False)
    parser.add_argument('--supervised_model_path', help='supervised-model log path', default="/mnt/research/judy/reward_shaping/sanity/supervised-atari-{}", type=str)
    parser.add_argument('--train_epochs', help='Number of epoches to train using expert data', default=1000, type=int)
    parser.add_argument('--expert_model_path', help='Model used to evaluate expert demonstration', default=None)
    parser.add_argument('--expert_data_id', help='file id to save expert data', default=1, type=int)
    parser.add_argument('--expert_episode', help='number of expert episodes to eval', default=10, type=int)
    parser.add_argument('--expert_save', help='If save episodes', default=False, type=bool)
    parser.add_argument('--render', help='If render the environment', default=False, type=bool)


    args = parser.parse_args()

    ENV_NAME = args.env
    NUM_ACTIONS = get_player().action_space.n
    logger.info("Environment: {}, number of actions: {}".format(ENV_NAME, NUM_ACTIONS))

    args.supervised_model_path = args.supervised_model_path.format(ENV_NAME)
    if not args.expert_model_path:
        args.expert_model_path = settings.supervised_model_checkpoint[args.env] 

    if args.task == 'eval':
        generate_expert_demonstration(args)

    elif args.task == "train":
        logger.info("Logger/Model Path: {}".format(args.supervised_model_path))
        train(args)
