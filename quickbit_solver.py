# Include Debug and System Tools
import traceback
import sys, os, os.path

import json
from datetime import datetime, timedelta, time, date

import random

import torch
from torch.utils.data import Dataset, DataLoader

from QuickBitsNN.data_io.const import INCOMPLETE_DATA_FILE_PATH

from QuickBitsNN.framework.quickbit import QuickBit, QuickBitsDataCart
from QuickBitsNN.framework.checkpoint import CheckPoint


from QuickBitsNN.transformer.train import RunModelTrain, LoadCheckpoint, LoadSavedCheckpoint

from QuickBitsNN.data_io.sort import SortQuickBits, find_files_with_similar_names
from QuickBitsNN.data_io.sniff import SniffQuickBits
from QuickBitsNN.data_io.show import ShowLastQuickBit, PlotProbabilityMapsNonceBitsTime, MeanState
from QuickBitsNN.data_io.load import LoadQuickBits


from QuickBitsNN.environment.env import GameWrapperEnvironmentQuickBits


#----------------------------------------------------------------
#----------------------------------------------------------------
def StartSniff():
    resume_quickbit_hash = "00000000000000846fad66482b46b5fb995013551fb08ac1a027f0e3e0346c0f"
    direction = 'forward'
    SniffQuickBits(resume_quickbit_hash, direction=direction, batchsniff=37)

#----------------------------------------------------------------
def StartSort():
    files_to_bin = find_files_with_similar_names( INCOMPLETE_DATA_FILE_PATH, f'quickbits_batchsniff_')
    for i in files_to_bin:
        quickbits = LoadQuickBits( i )
        SortQuickBits( quickbits )
        print(f"File {i} Bin Complete.")
    print(f"Done!")
    exit()

#----------------------------------------------------------------
def StartShow():

    MeanState()
    exit()
    all_quickbits = []
    files_to_bin = find_files_with_similar_names( INCOMPLETE_DATA_FILE_PATH, f'quickbits_coresniff_')
    files_to_bin = sorted(files_to_bin)
 
    print(files_to_bin)
    for i in files_to_bin:
        all_quickbits += LoadQuickBits( f"{i}" )
    
    validation_quickbits = LoadQuickBits( f"quickbits_validation_coresniff_forward_8_(11271).json" )
    print(f"Total QuickBits = {len(all_quickbits)}.")
    PlotProbabilityMapsNonceBitsTime(all_quickbits, validation_quickbits )

    exit()

#----------------------------------------------------------------
#----------------------------------------------------------------
def StartTrain(model=f'🍩'):

    checkpoint = CheckPoint()
    checkpoint.label = "📏"
    checkpoint.mdl = "Torch_Transformer_mps_NoncelessHeader-to-Nonce"
    checkpoint.ckp = 5
    checkpoint.ver = 0


    training_quickbits = []
    files_to_bin = sorted(find_files_with_similar_names( INCOMPLETE_DATA_FILE_PATH, f'quickbits_coresniff'))

    for i in files_to_bin:
        training_quickbits += LoadQuickBits( f"{i}" )

    training_quickbits = training_quickbits[1:] #Skip over the first quickbit that does not have the prev_block (ZeroBlockHash)
    training_quickbits = sorted(training_quickbits, key=lambda x : x.time, reverse=False)
  
    validation_quickbits = LoadQuickBits( f"quickbits_validation_coresniff_forward_8_(11271).json" )

    if '📏' in model:
        random.shuffle(training_quickbits)
        random.shuffle(validation_quickbits)
        
        dataloader_cart = QuickBitsDataCart(train=None, validate=None, test=None)
        dataloader_cart.train  = training_quickbits
        dataloader_cart.validate = validation_quickbits
        dataloader_cart.test = validation_quickbits

        loaded_checkpoint = LoadSavedCheckpoint(f'{checkpoint.checkpoint_file()}')
        if loaded_checkpoint != None: checkpoint = loaded_checkpoint
        RunModelTrain(dataloader_cart, checkpoint)

    
    if '📏' not in model:
        random.shuffle(training_quickbits)

        environment = GameWrapperEnvironmentQuickBits()
        environment.start_game(checkpoint=checkpoint, n_training_steps=128, n_epidsodes=len(training_quickbits)//8, quickbits=training_quickbits, train=True, save_frames=False, save_gifs=False)
        
    



#----------------------------------------------------------------
#----------------------------------------------------------------
def main():

    StartTrain()
    #StartShow()
    #StartSort()
    #StartSniff()


#----------------------------------------------------------------
#----------------------------------------------------------------
if __name__ == '__main__':
    main()

