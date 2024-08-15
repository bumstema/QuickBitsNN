################################################################
import numpy as np
import torch
import os
from typing             import List, Dict
from dataclasses        import dataclass, asdict, field

from ..data_io.utils        import get_device
from ..data_io.const        import CHECKPOINT_FILE_PATH
from ..framework.constants  import BATCH_SAMPLES



#  Contains all the Parameters to Train the Neural Network
#--------------------------------------------------------------
#--------------------------------------------------------------
@dataclass
class CheckPoint:

    mdl: str
    label: str
    ver: int
    log_ver: int
    qb: int
    ckp: int
    epoch: int
    retrain: int
    steps_per_epoch: float
    decay_rate: float
    lr: float

    current_lr: float
    current_epoch: int
    min_loss: float
    weighted_loss: bool

    train: bool
    validate: bool
    deploy: bool

    exclude_encoder: bool

    notes: str

    config_mdl: Dict = field(default_factory=lambda: ({}))
    losses: Dict = field(default_factory=lambda: ({}))
    gpu_usage: Dict = field(default_factory=lambda: ({}))

    preblock_headers: Dict = field(default_factory=lambda: ({}))

    #  Default Values for Simulation
    mdl         = f''       # Model Name
    label       = f''       # Emoji used for Model
    ver         = 0         # Version Number
    log_ver     = 0         # Logger Version for Tensorboard
    qb          = 0         # Total QuickBits run through Model Training
    ckp         = 0         # Number of Checkpoints Completed
    epoch       = 110       # Total Epochs to Train for
    retrain     = 0         # Counter for Number of Restarted Trainings
    steps_per_epoch = 0     # Used for One Cycle Scheduler
    decay_rate  = 0.0123456  # Value for Optimizer Weight Decay
    lr          = 1.3953064179598688e-07 #0.01254   # Learning Rate  # (0.024 to 0.026)

    current_lr  = 0.0       # Last Learning Rate during Training
    current_epoch = 0       # Number of Epochs Trained so far
    min_loss    = 9e9       # Minimum Loss as Found during Training
    weighted_loss = True    # Flag to set CrossEntropyLoss weights

    train       = True      # Flag to Allow Model Training
    validate    = True      # Flag to Allow Model Validation
    deploy      = False     # Flag to Use Model for Inference

    exclude_encoder = True  # Flag to Select Transformer as Translation or Generation Mode.
    notes       = f''

    def code(self):
        return f'{self.label}_{self.mdl}_mdl.{self.ver:0>2}ver.{self.qb}qb.{self.epoch}e'

    def checkpoint_name(self):
        return f"{self.label}_{self.mdl}_mdl.{self.ver}ver.{self.ckp}ckp"

    def checkpoint_file(self, type=f''):
        return  f"QuickBitsNN_Checkpoint_{type}({self.checkpoint_name()})"

    def checkpoint_path_to_file(self, type=f''):
        return  f'{self.checkpoint_name()}/QuickBitsNN_Checkpoint_{type}({self.checkpoint_name()})'

    def checkpoint_path_to_save_file(self, type=f''):
        root_dir = f'{self.checkpoint_name()}/'
        type_dir = f'{type}/'
        ver_dir = f'ver_{self.log_ver}/'
        filename= f'/QuickBitsNN_{type}_({self.checkpoint_name()})'
        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir + type_dir)
        except: pass

        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir + type_dir + ver_dir )
        except: pass

        return  CHECKPOINT_FILE_PATH + root_dir + type_dir + ver_dir + filename


    def checkpoint_path_to_img(self, type=f'', epoch=0, batch=0):
        root_dir = f'{self.checkpoint_name()}/img/'
        ver_dir = f'ver_{self.log_ver}/'
        ep_dir = f'epoch_{epoch}/'
        b_dir = f'batch_{batch}/'

        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir)
        except: pass

        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir + ver_dir)
        except: pass

        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir + ver_dir + ep_dir)
        except: pass

        try: os.mkdir(CHECKPOINT_FILE_PATH + root_dir + ver_dir + ep_dir + b_dir)
        except: pass

        return root_dir + ver_dir + ep_dir + b_dir

    def log_path(self):
        return CHECKPOINT_FILE_PATH +self.checkpoint_name()+ f'/log'
        
    def update_quickbit_count(self, count=BATCH_SAMPLES):
        self.qb += count

    def update_checkpoint(self):
        self.retrain += 1

    def load_model(self):
        if self.qb == 0: return False
        return True

    def set_to_deploy(self):
        self.deploy = True
        self.train = False
        self.validate = False



    def make_checkpoint_dir(self):
        try:
            os.mkdir(CHECKPOINT_FILE_PATH + f'{self.checkpoint_name()}/')
            os.mkdir(CHECKPOINT_FILE_PATH + f'{self.checkpoint_path_to_img()}')
        except:
            pass
        return