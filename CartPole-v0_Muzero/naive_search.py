import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model

device = paddle.get_device()
dtype = 'float32'

# ALTERNATIVE FOR MCTS
def naive_search(agent, state, num_actions, gamma, n=3):
# search the max value prediction and add rewards along the way of fully expanded tree at depth n 
# return the first action from the root to the max value node (in combination with eps greedy policy)

    possible_actions = np.array(list(range(num_actions)))

    _, target_v = agent.prediction_model(state)
    
    v = None
    rewards = paddle.to_tensor([0.0],place=device)


    
    for depth in range(n):
        dim = state.shape[1]
        state = paddle.tile(state, repeat_times=[1, num_actions]).reshape([-1,dim]) 
        actions = np.repeat([possible_actions], (num_actions ** depth), axis=0).flatten() 
        actions = np.sort(actions) 

        state, _, v, reward = agent.rollout_step(state, actions)
        state, v, reward = state.detach(), v.detach(), reward.detach()

        
        # add reward with respect to tree path

        rewards = paddle.tile(rewards.reshape([-1,1]), repeat_times=[1,num_actions]).reshape([-1]) 
        
        #discount reward at depth
        reward = reward * (gamma ** depth) 
        rewards = rewards + reward
    
    v = v.numpy()
    rewards = rewards.numpy() 
    # discount value prediction
    v = v * (gamma ** n)
    # add rewards
    v = v + rewards 
    
    # max selection
    
    max_index = np.argmax(v) 
    indexes_per_action = num_actions ** (n-1)
    action = int(max_index/indexes_per_action) 
    
    return action, target_v   
    

    
    
    
    