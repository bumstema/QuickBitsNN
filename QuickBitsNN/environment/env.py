from dataclasses import dataclass
from datetime import datetime, timedelta, time, date
from collections import deque
import hashlib
import random
import torch
import numpy as np
import math

from .state import State
from .agent import QLearningAgent
from ..data_io.utils import save_json_file, load_json_file
from ..data_io.utils import unpack_nested_list,  progress
from ..data_io.const import CURRENT_DIR_PATH, NN_CHECKPOINTS_PATH


from ..framework.tokenizer import Tokenizer


        
        
#----------------------------------------------------------------
@dataclass
class GameWrapperEnvironmentQuickBits():
    cartridge_title = 'QUICKBITS'
    
    # -----------------------------------
    def __init__(self, *args, **kwargs):

        self.agent =  0  # Actor in the environment
        self.action = 0  # Button Input for game
        self.state = {}  # Result of Agent acting in the environment
        self.reward = 0  # Incentive towards certain actions
        self.done = False
        self.failed = False
        self.step = 0
        
        self.episode = 0
        self.episode_cumulative_reward = 0
        self.q_learning_stats = deque([], maxlen=4096)
        
        self.tokenizer = Tokenizer()
 
    

    # -----------------------------------
    def start_game(self,checkpoint=[], n_training_steps=1, n_epidsodes=1, train=True, quickbits=[], save_frames=False, save_gifs=True):

        self.frame_type='state'
        self.agent = QLearningAgent(frame_type=self.frame_type)
        
        self.ep_min = 0
        self.ep_max = n_epidsodes
        self.quickbits = quickbits
        self.checkpoint = checkpoint
        self.train = train
        self.n_training_steps = n_training_steps
        self.fastest_run_n_steps = 900
        self.agent.max_episode_steps = n_training_steps
        self.total_actions = 0
        self.total_rewards = 0
        self.n_logs = 0
        self.n_states_visited = 0
        self.zeros = 0
        
        self.current_level_label = f'quickbits'

        self.ChooChooseModelTrain(load_level='quickbits', load_episode=0)
       
        self.ResetGameLevel(new_level=f'quickbits')
        self.RunEpisodes()
        
       
            
    # -----------------------------------
    def ChooChooseModelTrain(self, load_level='level-1', load_episode=0):

        try:
            model_name = f'Qlearning_model_({load_level}).pth'
            self.agent.load_model(NN_CHECKPOINTS_PATH+f'{model_name}')
        
        except:
            print(f"Failed to load Qlearning model... ({model_name})")

        if self.train:
            try:
                self.q_learning_stats = load_json_file( NN_CHECKPOINTS_PATH + f'Qlearning_stats_(quickbits).json' )
            except:
                print(f"Starting new stats output file.")
                self.q_learning_stats = deque([], maxlen=4096)
           
           
           
           
    # -----------------------------------
    def ResetGameLevel(self, new_level=f'level-1', save_state_file=f''):
    
        self.current_level_label = new_level
        self.state = State()
        self.state.done=True
        self.reward=-1
        self.state = self.update_state()
        self.next_state = self.state
        self.agent.ResetMemory()
 

    
    # -----------------------------------
    def RunEpisodes(self,):
    
        for episode in range(self.ep_min, self.ep_max):

            # Reset variables each new episode
            self.episode = episode
            self.agent.episode = episode

            # Define the First State
            self.action = [0,0,0,0]

            self.agent.n_actions_taken_by_Q = 0
            self.total_actions = 0
            self.n_states_visited = 0
            self.loss = 0
            self.q = 0
            self.episode_cumulative_reward = 0
            self.start_time = datetime.now()
            

            
            ################
            self.Stepisode()
            ################
            
            if (self.episode % 16 == 0) | (self.episode >= self.ep_max-1) :
                self.show_episode_stats()
            
            
            if self.done | self.failed:
                self.agent.update_cache_for_fastest_runs()
                
                #  - (6) - Log Parameters for Evaluation
                self.log()
                
                self.done = False
                self.failed = False
                
                
        return
            
            
            
            
            
    # -----------------------------------
    def Stepisode(self,):
        step = 0
        #for index, step in enumerate(range(0, self.n_training_steps)):
        while (not self.done):
            step += 1
            self.step = step
            self.agent.episode_steps = step
           
             
            #  - (1) - Agent takes action
            self.action =  self.agent.select_action(self.state)
            
            
            #  - (2) - Pass action into environment
            self.reward, self.done = self.state.Reward( self.action )
            self.episode_cumulative_reward += self.state.reward
            self.zeros = self.state.zeros
            
            
            #  - (3) - Environment changes states
            
            self.failed = False
            if (self.step+2 > self.n_training_steps):
                #self.final_state = self.state
                #self.show_episode_stats()
                self.failed = True

            self.next_state = self.update_state()

    
            #  - (4) - Cache system and response
            self.agent.cache(self.state, self.action, self.reward, self.next_state, self.done)


            #  - (5) - Apply updates to Q_learning model
            if self.train:
                _q, _loss = self.agent.learn()
                if _q is not None:
                    self.q += _q
                    self.loss += _loss
                            
 
            #  - (6) - Advance into next state
            
            if self.done:
                self.final_state = self.state
            self.state = self.next_state
   
   
         
            #if self.failed: break
            if self.done: break
            



    # -----------------------------------
    def show_episode_stats(self):
        newline = ''
        if self.done: newline = f'\n'
        self.prefix = f""
        self.suffix = f"|| Done:({self.done})| steps: {self.step:^4} | states: {self.n_states_visited:^4} | T-R: {self.episode_cumulative_reward:>4} -{self.final_state.zeros:>4} ={self.episode_cumulative_reward - self.final_state.zeros:>3} | q: {self.q:6.3f} | L: {self.loss:6.3f} |{self.agent.episode_update_text} {newline}"
        progress( self.episode, self.ep_max, prefix=f'{self.prefix}', suffix=f'{self.suffix}')


    # -----------------------------------
    def log(self):
    
        if (self.episode == 0):
            save_json_file({f'QNetworkDCN': str(self.agent.q_network)}, NN_CHECKPOINTS_PATH + f'QNetworkDCN_model.json' )
    
        
        self.q_learning_stats.append({
            'episode':self.episode,
            'state':self.final_state.block_tokens[:-4].tolist(),
            'action':self.final_state.action.tolist(),
            'reward':self.final_state.reward,
            'zeros':self.final_state.zeros,
            'total_reward':self.episode_cumulative_reward,
            'loss':self.loss,
            'q':self.q,
            'steps':self.step,
            'n_states_visited':self.n_states_visited,
            'n_actions_taken_by_Q':self.agent.n_actions_taken_by_Q,
            'done':self.done,
            'failed':self.failed,
            'time':str(datetime.now()-self.start_time)
        })
        
        if  ((self.episode % self.agent.burnin == 0) & (self.episode != 0)) | (self.episode+1 == self.ep_max)  :
            save_json_file(list(self.q_learning_stats), NN_CHECKPOINTS_PATH + f'Qlearning_stats_({self.current_level_label}_{self.n_logs}).json' )
            self.n_logs += 1
            self.q_learning_stats = deque([], maxlen=self.agent.max_mem)
            self.agent.save_model(NN_CHECKPOINTS_PATH+f'Qlearning_model_({self.current_level_label}).pth')
            

    # -----------------------------------
    def update_state(self):
    

        # Move to Next Block When Nonce is Solved
        if self.state.done | self.failed:
            next = 1
            if self.failed: next = 0
            qb = self.quickbits[ self.episode + next ]
            # Build Next State
            self.n_states_visited += 1
            state_frames = qb.build_header(decode=True)
            state_frames = self.tokenizer.tokenize(state_frames)
            next_state = State(block=state_frames, bits=qb.bits, hash=qb.hash, nonce=qb.nonce_hex_string())
            return next_state


        # Explore New State if Zero Reward is solved.
        # (Simulates New Transactions or Permutation of List)
        if (not self.state.done) & (self.reward == 0):
            next_state = self.state
            next_state.block_tokens = next_state.RandomizeTime()
            self.n_states_visited += 1
            return next_state
            

        if (not self.state.done):
            return self.state
        


        # Continue Solving Nonce - Gradient Descent Chain
        if (not self.state.done) & (self.reward > 0) :
            return self.state
        

        # Explore New State if Zero Reward is solved.
        # (Simulates New Transactions or Permutation of List)
        if (not self.state.done) & (self.reward == 0):
            next_state = self.state
            next_state.block_tokens = next_state.RandomizeTokens()
            self.n_states_visited += 1
            return next_state




