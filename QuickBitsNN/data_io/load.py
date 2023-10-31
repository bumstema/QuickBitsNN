import numpy as np
import torch
import random
import json

from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta, time, date

from ..framework.quickbit import QuickBit
from ..framework.quickbit import load_quickbits, QuickBitsModelDataLoader, QuickBitsTransformerDataset
from ..data_io.const import INCOMPLETE_DATA_FILE_PATH
from ..data_io.utils import get_device
from ..framework.constants import BATCH_SIZE, BATCH_SAMPLES, VALIDATION_SAMPLES, TEST_SAMPLES
from ..framework.tokenizer import Tokenizer



#--------------------------------------------------------------
#--------------------------------------------------------------
def save_json_file(data, filename):
    serialized_data = data
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__, ensure_ascii=False)

#--------------------------------------------------------------
def load_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data


#  Sanity Check on Imported Data (hashes can be determined)
#--------------------------------------------------------------
def sanity_check_on_data( quickbits, quiet=True ):

    print(f"[🛃]  Sanity Check - Calculating Hash from QuickBit data ....")
    qb =  quickbits[-1]
    qb_z = quickbits[0]
    
    if not quiet:
        print(f"\tdate[0] : {datetime.fromtimestamp(qb_z.time)} \t date[-1] : {datetime.fromtimestamp(qb.time)}")
        print(f"\t{qb.ver = }\n\t{qb.prev_block = }\n\t{qb.mrkl_root  = }\n\t{qb.time  = }\n\t{qb.bits  = }\n\t{qb.nonce = }")
        print(f"\t{qb.build_header().decode('utf-8') = }")
        print(f"\tVerifying Hash Calculation from Raw Data...")
        print(f"\tKnown Hash    : {qb_hash} \t {len(qb_hash)} \n\tCalc. Hash    : {qb_calc_hash} \t {len(qb_calc_hash)}")
    
    qb_hash = quickbits[-1].hash
    qb_calc_hash = quickbits[-1].calculate_mining_hash(quickbits[-1].nonce)
    
    if qb_hash == qb_calc_hash :
        print(f"[✔︎] ....  Hashes Match: {qb_hash == qb_calc_hash}.  [✔︎]\n")
        return
        
    else:
        print(f"[✘] ....  Hashes DO NOT match -- quitting training step. [✘]")
        exit()


#  Combine Representations of the Training Data
#--------------------------------------------------------------
def LoadDataLoaderWithData(quickbits, checkpoint, hash_lookup={}, nonce_only=False):
    if hash_lookup == {}: hash_lookup = load_json_file(INCOMPLETE_DATA_FILE_PATH + f'quickbits_preblock_headers.json' )

    preblocks = []
    targets = []
    tokenizer = Tokenizer()
    
    # Current Nonceless Preblock Header to Nonce
    if nonce_only:
        preblocks = [ tokenizer.tokenize( f"{bit.build_nonceless_header(decode=True)}", end_token=False  ) for bit in quickbits ]
        targets = [ tokenizer.tokenize( f"{bit.build_header_nonce(decode=True)}", start_token=True ) for bit in quickbits ]
        
    # Previous Preblock Header to Current Preblock Header
    if not nonce_only:
        preblocks = [ tokenizer.tokenize( hash_lookup[bit.prev_block], end_token=False  ) for bit in quickbits ]
        targets = [ tokenizer.tokenize( f"{bit.build_header(decode=True)}", start_token=False ) for bit in quickbits ]
    
    preblocks = torch.stack(preblocks)
    targets = torch.stack(targets)
    targets = targets.requires_grad_(requires_grad=False)
    
    loaded_dataset = QuickBitsTransformerDataset(preblocks.to(get_device()), targets.to(get_device()))
    
    full_dataloader = DataLoader(loaded_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    return full_dataloader

    


#  Wrapper Function for Display Infor with Loading QuickBit Data
#--------------------------------------------------------------
def LoadQuickBits( data_file_name ):
    print(f"[📦]  Loading QuickBits ....")
    print(f"\tFile: \"{data_file_name}\"")
    quickbits  =  load_quickbits(INCOMPLETE_DATA_FILE_PATH + f'{data_file_name}')
    print(f"\tTotal QuickBits in File: ({len(quickbits)})")
    print(f"[✔︎] ....  Complete! QuickBits Loaded. [✔︎]\n")

    sanity_check_on_data( quickbits )

    return quickbits




#----------------------------------------------------------------
#----------------------------------------------------------------
def main():
    quickbits = LoadQuickBits()


#----------------------------------------------------------------
if __name__ == '__main__':
    main()


