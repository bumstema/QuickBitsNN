import sys, os, os.path
import numpy as np
import hashlib
import random
import torch
import time
import cv2
import math
import time
from PIL import Image, ImageFilter, ImageOps, ImageChops, GifImagePlugin
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import deque
from scipy.special import softmax

import torchvision.transforms as transforms

from ..data_io.utils import  unpack_nested_list, get_device, load_json_file


from ..framework.tokenizer import Tokenizer
from ..framework.functions import BitsZeros, HeaderNonceMiningHash, LeadingZeros
from ..framework.functions import little_endian, four_byte_hex


data = load_json_file( os.getcwd() + '/data/Average_QuickBit_Tokens.json')
PREBLOCK_TOKEN_AVERAGES = data['mean_tokens']
PREBLOCK_TOKEN_STD = data['std_tokens']


#----------------------------------------------------------------
#----------------------------------------------------------------
@dataclass
class State():

    # -----------------------------------
    def __init__(self, block=[], bits=0, hash=f'', nonce='',  *args, **kwargs):
        self.tokenizer = Tokenizer()
        self.block_tokens = block
        self.colour_frame = []
        self.bits = bits
        self.zeros = BitsZeros(str(self.bits))
        self.hash = hash
        self.nonce = nonce
        if self.nonce != '':
            #self.target_nonce_tokens = self.tokenizer.tokenize(little_endian(four_byte_hex(int(nonce,16))).decode('utf-8'))
            self.target_nonce_tokens = self.tokenizer.tokenize(str(self.nonce))

        self.bits_tokens =  self.tokenizer.tokenize(little_endian(four_byte_hex(int(self.bits))).decode('utf-8'))
        self.action = torch.randint(0, 256, (4,))
        self.reward = 0
        self.done = False
        self.known_solution = True
        
        
    # -----------------------------------
    def Reward(self, action_pixel):
        
        self.action = action_pixel
        
        preblock = self.tokenizer.detokenize(self.block_tokens[:-4])
        nonce = self.tokenizer.detokenize(self.action)
        hash = HeaderNonceMiningHash(preblock, nonce)
        
        self.reward = int(LeadingZeros(hash))
        
        self.done = False
        if self.reward >= self.zeros : self.done = True

        return  self.reward, self.done



    # -----------------------------------
    def RandomizeMerkleRoot(self):
        random_tokens = torch.randint(0, 256, (32,))
        start_position = 4+32
        updated_block = self.block_tokens
        updated_block[start_position:start_position+32] = random_tokens[:]
        self.known_solution = False
        return updated_block
        
    # -----------------------------------
    def RandomizeTime(self):
        # Pick a random time between Now and the Future
        unix_atm = int(time.time())
        unix_last_mined = 5390942400
        unix_one_year   =   31536000
        unix_one_month  =    2628000
        unix_one_week   =     604800
        unix_five_days  =     432000
        unix_three_days =     259200
        unix_two_days   =     172800
        unix_ten_mins   =        600
        
        #its_been = (unix_one_week) + unix_five_days + unix_three_days - unix_ten_mins + unix_two_days
        random_tokens = torch.randint(unix_atm, (unix_atm+(2*unix_one_week)+unix_ten_mins+1), (1,))
        #random_tokens = self.tokenizer.tokenize(hex(random_tokens.item())[2:])
        random_tokens = self.tokenizer.tokenize(little_endian(four_byte_hex(int(random_tokens.item()))).decode('utf-8'))
        updated_block = self.block_tokens
        updated_block[-12:-8] = random_tokens[:]
        self.known_solution = False
        return updated_block
        
    # -----------------------------------
    def RandomizeTokens(self):

        random_tokens = torch.normal(PREBLOCK_TOKEN_AVERAGES, PREBLOCK_TOKEN_STD)
        random_tokens = torch.remainder(random_tokens, 255)
        random_tokens[-8:-4] = self.bits_tokens
        return random_tokens.to(dtype=torch.long)
    
    # -----------------------------------
    def render(self):
        pixel_values = self.block_tokens
        pixel_values[-4:] = self.action
        pixel_values = pixel_values
        pixel_values = torch.t(pixel_values.reshape(-1,4)).unsqueeze(1)
        pixel_values = pixel_values.permute(1,2,0)
        return pixel_values.numpy()

    # -----------------------------------
    def tokenized_word_to_RGBA(self):
        pixel_values = self.block_tokens
        pixel_values[-4:] = self.action
        pixel_values = pixel_values / 255
        pixel_values = torch.t(pixel_values.reshape(-1,4)).unsqueeze(1)
        return pixel_values
    
   
    # -----------------------------------
    def save_colour_frame(self):
        for idx, frame in enumerate(self.colour_frame):
            with Image.fromarray((frame).astype(np.uint8)) as pic:
                pic.convert("RGB").save(NN_TRAINING_PATH+f'{self.frame_labels[idx]}.png')

    # -----------------------------------
    def tensorize(self, frame_type='' ):
        return self.tokenized_word_to_RGBA()
    
    # -----------------------------------
    def normalize_pixels_around_zero(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        norm_frame = img - mean_val
        if std_val == 0 : std_val = 1
        norm_frame = norm_frame / std_val
        return norm_frame.astype(np.float32)
    
    # -----------------------------------
    def normalize_pixels_to_mid_RGB(self, img):
        mean_val = np.mean(img)
        std_val = np.std(img)
        diff_frame = img - mean_val
        max_pixel_val = np.max(diff_frame)
        if max_pixel_val == 0 : max_pixel_val = 1
        diff_frame = ((diff_frame / (max_pixel_val)) * 255//2) + (255//2)
        return diff_frame.astype(np.uint8)
        
    # -----------------------------------
    def normalize_pixels_between_zero_and_one(self, img):
        max_val = 255
        norm_frame = img / max_val
        return norm_frame.astype(np.float32)
