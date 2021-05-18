import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from Networks import Representation_Model, Dynamics_Model, Prediction_Model
from naive_search import naive_search


device = paddle.get_device()
dtype = 'float32'

class MuZero_Agent(paddle.nn.Layer):
    
    def __init__(self, num_actions, representation_model, dynamics_model, prediction_model, eps=0.5, gamma=0.99):
        super().__init__()
        
        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.prediction_model = prediction_model
        
        self.num_actions = num_actions
  
        self.eps = eps
        self.gamma = gamma
        
    # def forward(self, obs):
    #     pass
            
    def naive_search_inference(self, obs): # inference with naive_search
    
        start_state = self.representation_model(obs)
        action, v = naive_search(self, start_state, self.num_actions, self.gamma)
        
        # use epsilon greedy policy instead of everytime taking max action
        greedy = paddle.rand([1])
        if self.eps > greedy: # random
            return np.random.choice(self.num_actions, 1)[0], v
        else: # max
            return action, v
        
    def inital_step(self, obs):
    # first step of rollout for optimization
        state = self.representation_model(obs)
        p, v = self.prediction_model(state)
        
        return state, p, v
       
        
    def rollout_step(self, state, action): 
    # unroll a step
    
        batch_size = state.shape[0]
        
        action_encoding = paddle.to_tensor(action,dtype= dtype, place=device)
        action_encoding = paddle.reshape(action_encoding,[batch_size,1]) / self.num_actions
        in_dynamics = paddle.concat(x = [state,action_encoding],axis=1)
        
        next_state, reward = self.dynamics_model(in_dynamics)
        p, v = self.prediction_model(next_state)

        return next_state, p, v, reward
    