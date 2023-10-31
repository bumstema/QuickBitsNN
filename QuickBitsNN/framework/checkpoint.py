################################################################
import numpy as np
import torch

from typing             import List, Dict
from dataclasses        import dataclass, asdict, field

from ..data_io.utils        import get_device
from ..framework.constants  import BATCH_SAMPLES



#  Contains all the Parameters to Train the Neural Network
#--------------------------------------------------------------
#--------------------------------------------------------------
@dataclass
class CheckPoint:

    qb:      int
    ver:     int
    ckp:     int
    epoch:   int
    retrain: int

    mdl: str
    label: str
    
    lr: float
    min_loss: float
    zeros: int

    train: bool
    validate: bool
    deploy: bool
    
    config_mdl: Dict = field(default_factory=lambda: ({}))
    losses:     Dict = field(default_factory=lambda: ({}))
    gpu_usage:  Dict = field(default_factory=lambda: ({}))


    #  Default Values for Simulation
    qb          = 0
    ver         = 0
    ckp         = 0
    epoch       = 128
    retrain     = 0

    mdl         = f""
    label       = f""
    
    lr          = 1.33e-7
    zeros       = 0
    min_loss    = 9e9

    train       = True
    validate    = True
    deploy      = True


    def code(self):
        return f'{self.label}_{self.mdl}_mdl.({self.zeros})0s.{self.ver:0>2}v.{self.qb}qb.{self.epoch}e'

    def checkpoint_name(self):
        return f"{self.label}_{self.mdl}_mdl.({self.zeros})0s.{self.ckp}ckp."

    def checkpoint_file(self):
        return  f"QuickBitsNN_Checkpoint_({self.checkpoint_name()})"

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
