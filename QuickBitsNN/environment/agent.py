# Include Debug and System Tools
import traceback
import sys, os, os.path
import transformers
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import namedtuple, deque
import random
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Function
import torchvision.models as models
import torchvision


from ..framework.tokenizer import Tokenizer
from ..framework.functions import BitsZeros, LeadingZeros
from ..data_io.utils import  unpack_nested_list, get_device
from .dcn_model import DeepConvNet



#----------------------------------------------------------------
#----------------------------------------------------------------
class QLearningAgent:

    # -----------------------------------
    def __init__(self, learning_rate=1.33e-7, gamma=0.95, frame_type=f'state' ):
        self.type=frame_type
        self.channels=4
        self.q_network = DeepConvNet(init_c=self.channels, enc_sizes=[64,128,256,512], dec_sizes=[512])
        print(self.q_network)
        self.loss_fn = nn.HuberLoss()
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate, momentum=0.95, weight_decay=0.0001234 )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 512)
        self.discount_factor_gamma = gamma
    
        self.batch_size = 32
    
        self.max_mem = 4096*8
        self.memory = deque([], maxlen=self.max_mem)

        self.burnin = 1024   # min. experiences before training
        self.learn_every = 128  # no. of experiences between updates to Q_online
        self.sync_every = self.max_mem//16  # no. of experiences between Q_target & Q_online sync
        self.sync_bin = 0

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9999775
        self.exploration_rate_min = 0.1
        self.n_actions_taken_by_Q = 0

        self.episode = 0
        self.episode_steps = 0
        self.max_episode_steps = 99999
        
        self.frame_type=f''
        self.episode_update_text =f''
        self.tokenizer = Tokenizer()
        
    
    # -----------------------------------
    def ResetMemory(self):
        del self.memory
        self.memory = deque([], maxlen=self.max_mem)
        

    # -----------------------------------
    def select_action(self, state):
    
        solve_cond = (self.episode_steps >= self.max_episode_steps)
    
            # (Directed) # Previously Solved Nonce = action index
        if solve_cond :
            return state.target_nonce_tokens.to(device=get_device())
        

        # (Explore) # Select a random action
        if (not solve_cond) & (random.random() < self.exploration_rate):
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
            # (Explore) # Random Nonce = action index
            return torch.randint(0, 256, (4,), requires_grad=False, device=get_device())
        
    

        # (Exploit) # Use NN to solve best action
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = state.tensorize(frame_type=self.type).to(get_device()).unsqueeze(0)
            action_values = self.q_network(state_tensor, q_model="online")
            action_index = action_values.argmax(dim=-1)[0]
        
        self.n_actions_taken_by_Q += 1

        return action_index





    # -----------------------------------
    def cache(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_mem: self.memory.popleft()
        self.memory.append({"state": state, "action": action, "reward": reward, "next_state": next_state, "done": done})
        return

            
    # -----------------------------------
    def recall(self):

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*[[b.get(key) for key in ["state", "action", "reward", "next_state", "done"]] for b in batch])
        
        state = [state_.tensorize(frame_type=self.type) for state_ in list(state)]
        state = torch.stack(state).to(device=get_device())
        
        next_state = [state_.tensorize(frame_type=self.type) for state_ in list(next_state)]
        next_state = torch.stack(next_state).to(device=get_device())
  
        action =  torch.stack(list(action)).to(device=get_device())
        reward = torch.tensor(list(reward), device=get_device())
        done = torch.tensor(list(done), device=get_device())

        return state, action, reward, next_state, done


    # -----------------------------------
    def update_cache_for_fastest_runs(self):

        self.synced = False
        self.sync_bin += self.episode_steps
        if  self.sync_bin >  self.sync_every:
            self.sync_bin = 0
            self.synced = True
            self.sync_Q_target()
    
        self.episode_update_text = f"| Q_p: {self.n_actions_taken_by_Q:^4}| Explorate: {self.exploration_rate:.4f} | Mem: {len(self.memory):^6} | Sync: {self.synced} ||"

      
    # -----------------------------------
    def learn(self):

        if len(self.memory) < self.burnin:
            return None, None

        if len(self.memory) == 0:
             return None, None

        if ( ((self.episode * self.max_episode_steps) + self.episode_steps) % self.learn_every) != 0:
            return None, None

        # Sample from memory
        state, action, reward, next_state, done = self.recall()

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        
        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)
        
        q = td_est.mean().item()
        
        return q, loss


    # -----------------------------------
    #@torch.no_grad()
    def td_target(self, reward, next_state, done):
        self.q_network.eval()
        
        # Q_target(r,s,a)
        with torch.no_grad():
            next_state_Q = self.q_network(next_state, q_model="online")
            best_action = torch.argmax(next_state_Q, axis=-1)
            next_Q = self.q_network(next_state, q_model="target")
            next_Q = next_Q[ np.arange(0, self.batch_size), :, best_action[:, -1] ]
    
        return (reward.unsqueeze(1) + (1 - done.unsqueeze(1).float()) * self.discount_factor_gamma * next_Q).float()


    # -----------------------------------
    def td_estimate(self, state, action):
        self.q_network.train()
        
        # Q_online(s,a)
        current_Q = self.q_network(state, q_model="online")
        current_Q = current_Q[ np.arange(0, self.batch_size), :, action[:, -1]  ]
        return current_Q

    # -----------------------------------
    def update_Q_online(self, td_estimate, td_target):
        loss = 0 
        self.q_network.train()
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step( self.episode + self.episode_steps  / self.max_episode_steps )

        return loss.item()
  
    # -----------------------------------
    def sync_Q_target(self):
        self.q_network.target.load_state_dict(self.q_network.online.state_dict())

    # -----------------------------------
    def sync_Q_online(self):
        self.q_network.online.load_state_dict(self.q_network.target.state_dict())

    # -----------------------------------
    def save_model(self, file_name_path_model_state_dict):
        torch.save(self.q_network.state_dict(), file_name_path_model_state_dict )

    # -----------------------------------
    def load_model(self, file_name_path_model_state_dict ):
        self.q_network.load_state_dict(torch.load(file_name_path_model_state_dict))

    # -----------------------------------
    # -----------------------------------

