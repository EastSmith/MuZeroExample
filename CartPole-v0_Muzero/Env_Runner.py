import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
import numpy as np
import time

device = paddle.get_device()
dtype = 'float32'

class Logger:
    
    def __init__(self, filename):
        self.filename = filename
        
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()
        
class Env_Runner:
    
    def __init__(self, env):
        super().__init__()
        
        self.env = env
        self.num_actions = self.env.action_space.n
        
        self.logger = Logger("episode_returns")
        self.logger.log("training_step, return")
        
        self.ob = self.env.reset()
        self.total_eps = 0
        
    def run(self, agent, render=False):
        
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vs = []
        
        self.ob = self.env.reset()
        self.obs.append(paddle.to_tensor(self.ob))
        
        done = False
        while not done:
            
            action, v = agent.naive_search_inference(paddle.to_tensor(self.ob,dtype= dtype, place=device))
            
            self.ob, r, done, info = self.env.step(action)
            
            self.obs.append(paddle.to_tensor(self.ob))
            self.actions.append(action)
            self.rewards.append(paddle.to_tensor(r))
            self.dones.append(done)
            self.vs.append(v)
            
            if done: # environment reset
                if "return" in info:
                    self.logger.log(f'{self.total_eps},{info["return"]}')
                    print("Return:",info["return"])
            
            if render:
                self.env.render()
                time.sleep(0.024)
        
        self.total_eps += 1
                                    
        return self.make_trajectory()
        
        
        
    def make_trajectory(self):
        traj = {}
        traj["obs"] = self.obs
        traj["actions"] = self.actions
        traj["rewards"] = self.rewards
        traj["dones"] = self.dones
        traj["vs"] = self.vs
        traj["length"] = len(self.obs)
        return traj
        
        
        