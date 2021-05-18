import argparse
import numpy as np
import gym
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import time
import random
import os

from Env_Runner import Env_Runner
from Env_Wrapper import Env_Wrapper
from Agent import MuZero_Agent
from Networks import Representation_Model, Dynamics_Model, Prediction_Model
from Experience_Replay import Experience_Replay

device = paddle.get_device()
dtype = 'float32'

if __name__ == "__main__":
    
    filename = "model_cartpole_3_history_200_return.pdparams"
    agent = paddle.load(filename)
    agent.eps = 0.1
    
    raw_env = gym.make('CartPole-v0')
    env = Env_Wrapper(raw_env, 3)
    runner = Env_Runner(env)
       
    for i in range(100):
        _ = runner.run(agent, render=True)