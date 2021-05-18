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

def train():
    
    history_length = 3
    num_hidden = 50
    replay_capacity = 10000
    batch_size = 64
    k = 3
    n = 10
    lr = 2.5e-3
    
    start_eps = 1
    final_eps = 0.1
    final_episode = 1000
    eps_interval = start_eps-final_eps
    
    raw_env = gym.make('CartPole-v0')  

    num_obs_space = raw_env.observation_space.shape[0] 
    num_actions = raw_env.action_space.n  
    num_in = history_length * num_obs_space # history * ( obs )
    
    env = Env_Wrapper(raw_env, history_length)
    
    representation_model = Representation_Model(num_in, num_hidden)
    dynamics_model = Dynamics_Model(num_hidden, num_actions)
    prediction_model = Prediction_Model(num_hidden, num_actions)
    
    agent = MuZero_Agent(num_actions, representation_model, dynamics_model, prediction_model)

    runner = Env_Runner(env)
    replay = Experience_Replay(replay_capacity, num_actions)
    
    mse_loss = nn.MSELoss()
    
    optimizer = optim.Adam(learning_rate=lr,parameters=agent.parameters())  #  ,weight_decay=0.1
    # optimizer = optim.SGD(learning_rate=lr,parameters=agent.parameters())
 
    for episode in range(1500):#while True:
        
        agent.eps = np.maximum(final_eps, start_eps - ( eps_interval * episode/final_episode))
        
        # act and get data
        trajectory = runner.run(agent) 
        
        # save new data
        replay.insert([trajectory])
        
        #############
        # do update #
        #############
        
        if len(replay) < 10:
            continue
            
        for update in range(8):
            optimizer.clear_grad()
            
            # get data
            data = replay.get(batch_size,k,n)

            # network unroll data
            representation_in = paddle.stack([paddle.flatten(data[i]["obs"]) for i in range(batch_size)]) # flatten when insert into mem
            actions = np.stack([np.array(data[i]["actions"], dtype=np.int64) for i in range(batch_size)])
           
            def to_tensor(data):
                    data =[data[i].numpy() for i in range(len(data))]
                    data =paddle.squeeze(paddle.to_tensor(data,stop_gradient=True))
                    return data
  
            value_target = paddle.stack([to_tensor(data[i]["return"]) for i in range(batch_size)])
            rewards_target = paddle.stack([to_tensor(data[i]["rewards"]) for i in range(batch_size)])
            
            # loss
            
            loss = paddle.to_tensor(0,dtype= dtype, place=device)
            
            # agent inital step
            representation_in = paddle.cast(representation_in, dtype='float32')
            state, p, v = agent.inital_step(representation_in)
            # value_target[:,0].stop_gradient =True
            # value_loss = mse_loss(v, paddle.to_tensor(value_target[:,0],stop_gradient =False))
            value_loss = mse_loss(v, paddle.squeeze(value_target[:,0].detach()))
             
            loss += value_loss

            # steps
            for step in range(1, k+1):
            
                # step
                step_action = actions[:,step - 1]
                state, p, v, rewards = agent.rollout_step(state, step_action)
                
                value_loss = mse_loss(v, value_target[:,step].detach())
                
                reward_loss = mse_loss(rewards, rewards_target[:,step-1].detach())
                
                
                # print(f'value: {value_loss} || reward: {reward_loss}')
                loss += (value_loss + reward_loss)
               
            loss.backward()
            optimizer.step()
            
            if (episode+1)%100== 0:
                model_state_dict = agent.state_dict()
                paddle.save(model_state_dict, f"CheckPoint/{episode+1}.pdparams")
                

if __name__ == "__main__":

    train()