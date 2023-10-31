import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import math
import random
import torchvision
import numpy as np
import transformers

from dataclasses import dataclass, field
from collections import namedtuple, deque

from ..data_io.utils import unpack_nested_list, get_device

#----------------------------------------------------------------
#----------------------------------------------------------------
class ResNet(torch.nn.Module):
    def __init__(self, module):
        super(ResNet, self).__init__()
        self.SkipConnection = module

    def forward(self, inputs):
        return inputs + self.SkipConnection(inputs)
        
    

#----------------------------------------------------------------
def conv_block(in_f, out_f, repeat_blocks=1, *args, **kwargs):
    """ Based on the BottleNeck function of ResNet50. """
    
    btlnck = (out_f//4)
    block =  [[ResNet(nn.Sequential(
        nn.Conv2d(in_f, btlnck, kernel_size=1, padding=0, bias=False, device=get_device()),
        nn.BatchNorm2d( btlnck, device=get_device()),
        nn.LeakyReLU(),
        nn.Conv2d( btlnck, btlnck, kernel_size=(1,3), padding=(0,1), bias=False, device=get_device()),
        nn.BatchNorm2d(btlnck, device=get_device()),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(btlnck, in_f, kernel_size=1, padding=0, bias=False, device=get_device()),
        nn.BatchNorm2d(in_f, device=get_device()),
        nn.LeakyReLU(inplace=True)
        )) ] for i in range(1, repeat_blocks)]
        
    return nn.Sequential( *unpack_nested_list( block ))


#----------------------------------------------------------------
class ResnetEncoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        
        # Recommended defaults:
        # enc_sizes = [64,128,256,512]
        
        n_layers =  [2, 3, 3, 2]
        
        self.feature_steps = [nn.Conv2d(in_f, out_f, kernel_size=(1,3), padding=(0,1), device=get_device(), bias=False) for idx, (in_f, out_f) in enumerate(zip(enc_sizes, enc_sizes[1:]))]
        
        self.resnet_blocks = [ResNet(conv_block(in_f, out_f, repeat_blocks=n_layers[idx] )) for idx, (in_f, out_f) in enumerate( zip(enc_sizes[1:], enc_sizes[1:]) )]
        
        self.max_pool_blocks = []  # [N, C, h, w] -> [N, C, 1, 20] -> [N, C, 1, 10]  etc.
        self.max_pool_blocks.append( torch.nn.AdaptiveAvgPool2d( (1,20)) )
        self.max_pool_blocks.append( torch.nn.AdaptiveAvgPool2d( (1,10)) )
        self.max_pool_blocks.append( torch.nn.AdaptiveAvgPool2d( (1,5)) )
        self.max_pool_blocks.append( torch.nn.AdaptiveAvgPool2d( (1,1)) )

        
        self.combined_blocks = zip(self.feature_steps, self.resnet_blocks, self.max_pool_blocks)
        self.ConvolutionBlocks = nn.Sequential( *unpack_nested_list(self.combined_blocks) )
        
        
    def forward(self, x):
        return self.ConvolutionBlocks(x)
        
#----------------------------------------------------------------
def decode_block(in_f, out_f, repeats=2):
    return nn.Sequential(*unpack_nested_list([[
        nn.Linear(in_f, out_f, device=get_device()),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p = 0.05)] for i in range(repeats)] ))

#----------------------------------------------------------------
def Auto_Encoder( in_features=1024, out_features=1024):
    return ResNet(nn.Sequential(
        nn.Linear( in_features, (in_features//4), device=get_device()),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(p = 0.25),
        nn.Linear( (in_features//4), out_features, device=get_device()),
        nn.LeakyReLU(inplace=True)
        ))

#----------------------------------------------------------------
def resnet_docoder( n_features=1024, repeat_blocks=2 ):
    return nn.Sequential(
        decode_block(n_features, n_features),
        ResNet(  *[ decode_block(n_features, n_features) for i in range(repeat_blocks) ])
        )

#----------------------------------------------------------------
class DeepDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.DecoderBlocks = ResNet( decode_block(dec_sizes[0], dec_sizes[0])  )
        self.LastLinear = nn.Linear(dec_sizes[-1], n_classes, device=get_device())
        
        
    def forward(self, x):
        return nn.Sequential(self.DecoderBlocks, self.LastLinear)(x)



#----------------------------------------------------------------
class ToLogits(nn.Module):
    def __init__(self):
        super(ToLogits, self).__init__()
        
    def forward(self, x):
        N, _ = x.shape
        return torch.reshape(x, (N, 4, -1))



#----------------------------------------------------------------
#----------------------------------------------------------------
class DeepConvNet(nn.Module):
    """ Deep Convolutional Network for Q learning. """
    
    def __init__(self, init_c=4, enc_sizes=[64,128,256,512], dec_sizes=[512,512], n_classes=1024):
        super().__init__(  )
        # set initial channels: init_c=3 for RGB images
        
        self.enc_sizes = [init_c, *enc_sizes]
        self.dec_sizes = [ *dec_sizes]
        
        self.online =   nn.Sequential(
            *[ ResnetEncoder(self.enc_sizes),
                nn.Flatten(),
                Auto_Encoder(in_features=self.enc_sizes[-1], out_features=self.dec_sizes[0] ),
                DeepDecoder(self.dec_sizes, n_classes),
                ToLogits()
            ] )
    
        # Initialization
        for params in self.online.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)
                
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    
        # Assign the Same Initial Model to Target Class
        self.target = copy.deepcopy(self.online)
 
        # Q_target parameters are frozen.
        for p in self.target.parameters(): p.requires_grad = False
            
            

    # -----------------------------------
    def forward(self, state_, q_model="online"):
    
        if q_model == "online":
            return self.online(state_)
            
        elif q_model == "target":
            return self.target(state_)
        
#----------------------------------------------------------------
#----------------------------------------------------------------





#  The default model:
"""
QNetworkDCN(
  (online): Sequential(
    (0): ResnetEncoder(
      (conv_blocks): Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ResNet(
          (module): Sequential(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.01)
            (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): LeakyReLU(negative_slope=0.01)
            (5): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ResNet(
          (module): Sequential(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.01)
            (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): LeakyReLU(negative_slope=0.01)
            (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ResNet(
          (module): Sequential(
            (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.01)
            (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (4): LeakyReLU(negative_slope=0.01)
            (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (8): AdaptiveAvgPool2d(output_size=2)
      )
    )
    (1): Flatten(start_dim=1, end_dim=-1)
    (2): DeepDecoder(
      (dec_blocks): Sequential(
        (0): Sequential(
          (0): Linear(in_features=512, out_features=128, bias=True)
          (1): LeakyReLU(negative_slope=0.01)
        )
      )
      (last): Linear(in_features=128, out_features=8, bias=True)
    )
  )
"""


