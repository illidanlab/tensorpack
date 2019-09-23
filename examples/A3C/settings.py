#!/usr/bin/env python
# -*- coding: utf-8 -*-

path_prefix = "/mnt/research/judy/reward_shaping" 
#path_prefix = "/mnt/research/illidan/judy/reward_shaping" 

expert_data_path = {
    "Pong-v0": "{}/expert_data/Pong-v0".format(path_prefix),
    "Gravitar-v0": "{}/expert_data/Gravitar-v0".format(path_prefix)
}

pretraind_model_path = {
    "Pong-v0": "{}/pretrained_models/Pong-v0.npz".format(path_prefix),
    "Gravitar-v0": "{}/pretrained_models/Gravitar-v0.tfmodel".format(path_prefix),
    "Alien-v0": "{}/pretrained_models/Alien-v0.tfmodel".format(path_prefix),
    "BankHeist-v0": "{}/pretrained_models/BankHeist-v0.tfmodel".format(path_prefix),
    "WizardOfWor-v0": "{}/pretrained_models/WizardOfWor-v0.tfmodel".format(path_prefix),
    "Zaxxon-v0": "{}/pretrained_models/Zaxxon-v0.tfmodel".format(path_prefix),
}

supervised_model_checkpoint = {
    "Pong-v0": "{}/sanity/model_checkpoint".format(path_prefix),
    "Gravitar-v0": "{}/sanity/Gravitar-v0/supervised_model_checkpoint".format(path_prefix)
}
