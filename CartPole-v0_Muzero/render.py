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
    

    raw_env = gym.make('CartPole-v0')
    filename = "model_cartpole_3_history_200_return.pdparams"

    history_length = 3
    num_hidden = 50
    
    num_obs_space = raw_env.observation_space.shape[0] 
    num_actions = raw_env.action_space.n  
    num_in = history_length * num_obs_space # history * ( obs )
    
    representation_model = Representation_Model(num_in, num_hidden)
    dynamics_model = Dynamics_Model(num_hidden, num_actions)
    prediction_model = Prediction_Model(num_hidden, num_actions)
    
    agent = MuZero_Agent(num_actions, representation_model, dynamics_model, prediction_model)    

    model_state_dict  = paddle.load(filename)
    agent.set_state_dict(model_state_dict )
    agent.eps = 0.1
    
    env = Env_Wrapper(raw_env, history_length)
    runner = Env_Runner(env)
       
    for i in range(100):
        _ = runner.run(agent, render=True)
