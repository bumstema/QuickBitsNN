
import sys, os, os.path

import json

import hashlib

import math
import numpy as np
from datetime import datetime, timedelta, time, date
from dataclasses import  asdict

import random


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from ..framework.checkpoint import CheckPoint
from ..framework.tokenizer import  Tokenizer
from ..framework.functions import BitsZeros, HeaderNonceMiningHash, LeadingZeros, little_endian_hex_to_int
from ..framework.constants import MAX_INT
from ..framework.constants import BATCH_SAMPLES, BATCH_SIZE, VALIDATION_SAMPLES, TEST_SAMPLES
from ..framework.constants import HASHEX_TO_INDEX, INDEX_TO_HASHEX, START_TOKEN, END_TOKEN, PADDING_TOKEN, START_TOKEN_INDEX, END_TOKEN_INDEX

from ..data_io.const import MODEL_FILE_PATH, CHECKPOINT_FILE_PATH, INCOMPLETE_DATA_FILE_PATH
from ..data_io.utils import progress, load_json_file, save_json_file, get_device
from ..data_io.load import LoadDataLoaderWithData

from ..transformer.torch_transformer import Torch_Transformer, ModelConfig


#----------------------------------------------------------------
#----------------------------------------------------------------
def time_since_start(start_time, epoch):
    epoch_time = datetime.now()
    time_delta_from_current_epoch   = epoch_time - start_time
    average_epoch_loop_time_delta   = (time_delta_from_current_epoch/(epoch+1))
    next_epoch_done_at              = start_time + (time_delta_from_current_epoch) + average_epoch_loop_time_delta

    epoch_time_str          = str(epoch_time - start_time).split('.', 2)[0]
    next_epoch_done_at_str  = (str(next_epoch_done_at).split('.', 2)[0]).split(' ', 2)[1]
    
    return epoch_time_str, next_epoch_done_at_str


#----------------------------------------------------------------
def ShowTrainTime( epoch, num_epochs, start_time, loss ):
    epoch_time = datetime.now()
    time_delta_from_current_epoch   = epoch_time - start_time
    average_epoch_loop_time_delta   = (time_delta_from_current_epoch/(epoch+1))
    next_epoch_done_at              = start_time + (time_delta_from_current_epoch) + average_epoch_loop_time_delta

    epoch_time_str          = str(epoch_time - start_time).split('.', 2)[0]
    next_epoch_done_at_str  = (str(next_epoch_done_at).split('.', 2)[0]).split(' ', 2)[1]
    
    if (type(loss) is tuple) :
        train_loss, validation_loss = loss
        print(f"[🛤 ] \tEpoch[{epoch+1: >4}/{num_epochs}]  T.Loss[{train_loss:.16f}]  V.Loss[{validation_loss:.16f}]  ⏳[{epoch_time_str}] 🕓[{next_epoch_done_at_str}]")
        
    else:
        print(f"[🛤 ] \tEpoch[{epoch+1: >4}/{num_epochs}]  Loss[{loss:.16f}]  ⏳[{epoch_time_str}] 🕓[{next_epoch_done_at_str}]")



#----------------------------------------------------------------
def LoadSavedModel( model_path_name, model_instance):

    file_name_path_model_state_dict = CHECKPOINT_FILE_PATH + f'{model_path_name}.pth'
    try:
        print(f"[⚙️ ]  Importing Model From File ....")
        print(f"\t\"{model_path_name}.pth\"")
        model_instance.load_state_dict(torch.load(file_name_path_model_state_dict))
        #model_instance.train()
        print(f"[✔︎] ....  was Successfully Loaded! [✔︎]")
    except:
        print(f"[✘] ....  has Failed to Load. [✘]")
        raise
        exit()

    return model_instance
    
    
#----------------------------------------------------------------
def LoadSavedCheckpoint(checkpoint_path_name):

    file_name_path_checkpoint = CHECKPOINT_FILE_PATH + f'{checkpoint_path_name}.json'
    try:
        print(f"[⚙️ ]  Importing Checkpoint From File ....")
        print(f"\t\"{checkpoint_path_name}.json\"")
        checkpoint = LoadCheckpoint(file_name_path_checkpoint)
        print(f"[✔︎] ....  was Successfully Loaded! [✔︎]")
        return checkpoint

    except:
        print(f"[✘] ....  No File to Load... New Checkpoint File Created After Training. [✘]")
        return None





#----------------------------------------------------------------
def ZerosFromPredictions(pred, current_block, tokenizer=Tokenizer()):
    batch, seq_len, vocab_prob = pred.shape
    softy_max_pred = torch.nn.functional.softmax(pred.detach(), dim=-1)
    next_items = torch.argmax(softy_max_pred[:, :4], dim=-1)
    generated_nonces = next_items.squeeze(-1)

    hash_zeros = [LeadingZeros(HeaderNonceMiningHash(
        tokenizer.detokenize(current_block[idx_, :-4]),
        tokenizer.detokenize(generated_nonces[idx_])))
        for idx_ in range(batch)]
    hash_zeros = torch.tensor(hash_zeros, device=get_device()).to(dtype=torch.float32)

    bits_zeros = [BitsZeros( little_endian_hex_to_int(tokenizer.detokenize(current_block[idx_, -8:-4]))) for idx_ in range(batch)]
    bits_zeros = torch.tensor(bits_zeros, device=get_device()).to(dtype=torch.float32)

    zero_score,zero_idx = torch.min( bits_zeros - hash_zeros, 0)
    zero_mean = torch.mean( hash_zeros.to(dtype=torch.float32) ).item()
    score = zero_score / bits_zeros[zero_idx]

    return score+zero_mean

    

#----------------------------------------------------------------
#----------------------------------------------------------------

def EvaluationStation(model, quickbits, checkpoint):
    """ Generates the next sequence tokens then compares how many match exactly. """
    
    total_items = 0
    actual_zeros = 0
    total_correct = 0
    tokenizer = Tokenizer()

    most_hash_zeros = {'zeros':0, 'hash':0, 'header':0, 'nonce':0}
    lest_zero_score = {'bits-zeros':9e9, 'hash':0, 'header':0, 'nonce':0}

    hash_lookup = load_json_file(INCOMPLETE_DATA_FILE_PATH + f'quickbits_preblock_headers.json' )



    print(f"\tMinibatch Carts: {len(quickbits)//TEST_SAMPLES}.")
    for i, mini_batch in enumerate(range(0, len(quickbits), TEST_SAMPLES)):
        dataloader = LoadDataLoaderWithData(quickbits[mini_batch:mini_batch+TEST_SAMPLES], checkpoint, hash_lookup=hash_lookup)
        
        model.eval()
        model.to(get_device())
        
        for previous_block, current_block in dataloader:
            batch, word_len = current_block.shape
            
            with torch.no_grad():
            
                pred = model(previous_block.unsqueeze(1), current_block[:,:-4].unsqueeze(1))
                softy_max_pred = torch.nn.functional.softmax(pred, dim=-1)
                next_items = torch.argmax(softy_max_pred[:,:4], dim=-1)
                generated_nonces = next_items.squeeze(-1)
                generated_words = generated_nonces


            hashes = [ HeaderNonceMiningHash(
                tokenizer.detokenize(current_block[idx_, :-4]),
                tokenizer.detokenize(generated_nonces[idx_, -4:]))
                for idx_ in range(batch)]

            hash_zeros = [LeadingZeros( hash_ ) for hash_ in hashes]
            hash_zeros = torch.tensor(hash_zeros, device=get_device()).to(dtype=torch.float32)

            bits_zeros = [BitsZeros(little_endian_hex_to_int(tokenizer.detokenize(current_block[idx_, -8:-4]))) for idx_
                          in range(batch)]
            bits_zeros = torch.tensor(bits_zeros, device=get_device()).to(dtype=torch.float32)


            bits_score, bits_idx = torch.min(bits_zeros - hash_zeros, 0)
            bits_score = bits_score.item()
            bits_idx = bits_idx.item()

            max_zero = torch.max(hash_zeros, 0)
            zeros = max_zero[0].item()
            zero_idx = max_zero[1].item()

            if zeros > most_hash_zeros['zeros']:
                preblock = tokenizer.detokenize(current_block[zero_idx, :])
                nonce = tokenizer.detokenize(generated_nonces[zero_idx, -4:])
                most_hash_zeros = {'zeros':zeros, 'hash':hashes[zero_idx], 'header':preblock, 'nonce':nonce}

            if bits_score < lest_zero_score['bits-zeros']:
                preblock = tokenizer.detokenize(current_block[bits_idx, :])
                nonce = tokenizer.detokenize(generated_nonces[bits_idx, -4:])
                lest_zero_score = {'bits-zeros':bits_score, 'hash':hashes[bits_idx], 'header':preblock, 'nonce':nonce}

            actual_zeros += sum(hash_zeros)
            total_items += len(hash_zeros)


            total_items += 4 * batch
            total_correct += torch.eq(generated_nonces.reshape(-1), current_block[:, -4:].reshape(-1)).sum().item()

            word = [tokenizer.detokenize(generated_words[index, -4:]) for index in [0, batch // 2, -1]]
            expected = [tokenizer.detokenize(current_block[index, -4:]) for index in [0, batch // 2, -1]]
            print(f'[ Batch:{i}  Gen:]\t \"{word[0]}\" \t \"{word[1]}\" \t \"{word[2]}\"')
            print(f'[ Batch:{i}  Exp:]\t \"{expected[0]}\" \t \"{expected[1]}\" \t \"{expected[2]}\"')
            print(f'-----------------')



    if total_items == 0: total_items = 1
    nonce_correct = {'correct': total_correct, 'hash leading zeros': actual_zeros, 'total': total_items,
                     '%': total_correct / total_items, 'zeros/item': actual_zeros / total_items}

    print(f"[🎟️ ]\tCorrect Predictions: {total_correct}/{total_items}  =  {total_correct / total_items:.4f}")
    print(f"[🎫️ ]\tLeading Zeros from Predictions: {actual_zeros}/{total_items}  =  {actual_zeros / total_items:.4f}")
    print(f"\t {most_hash_zeros = } ")
    print(f"\t {lest_zero_score = } ")

    nonce_correct.update({'most_hash_zeros':most_hash_zeros, 'lest_zero_score':lest_zero_score})

    return  nonce_correct
    
#----------------------------------------------------------------
def ValidationStation(model, loss_fn, quickbits, checkpoint):
    """ Evaluates the models' losses on the provided Training Data. """
    
    total_loss = 0
    total_items = 0
    total_batches = 0
    start_time = datetime.now()
    tokenizer = Tokenizer()
    hash_lookup = load_json_file(INCOMPLETE_DATA_FILE_PATH + f'quickbits_preblock_headers.json' )

  
    print(f"\tMinibatch Carts: {len(quickbits)//VALIDATION_SAMPLES}.")
    for i, mini_batch in enumerate(range(0, len(quickbits), VALIDATION_SAMPLES)):
        dataloader = LoadDataLoaderWithData(quickbits[mini_batch:mini_batch+VALIDATION_SAMPLES], checkpoint, hash_lookup=hash_lookup)
        
        model.eval()
        model.to(get_device())
        with torch.no_grad():
            for previous_block, current_block in dataloader:
                batch, word_len = current_block.shape
                pred = model(previous_block.unsqueeze(1), current_block[:,:-4].unsqueeze(1))
                loss = loss_fn(pred[:,-4:,:].reshape(-1, tokenizer.vocab_size), current_block[:, -4:].reshape(-1)) + ZerosFromPredictions(pred, current_block)

                total_loss += loss.detach().to(dtype=torch.float32).item()
                total_items += batch
        
        total_batches += 1
                
        epoch_time_str, next_epoch_done_at_str =  time_since_start(start_time, i)
        
        suffix = f"V.Loss/Item[{total_loss/total_items:.8f}]  V.Loss/Batch[{total_loss/total_batches:.8f}]  ⏳[{epoch_time_str}] 🕓[{next_epoch_done_at_str}]"
        progress(i + 1, -(-len(quickbits)//BATCH_SAMPLES), prefix=f"[🛤 ]", suffix=suffix)

    return total_loss / total_items
    
    
    
#----------------------------------------------------------------
def TrainStation(model, opt, sched, loss_fn, quickbits, checkpoint):
    """ Train the model over 1 epoch cycle. """

    total_loss = 0
    total_items = 0
    total_batches = 0
    lowest_loss = checkpoint.min_loss
    start_time = datetime.now()
    model_type = checkpoint.label
    tokenizer = Tokenizer()
    hash_lookup = load_json_file(INCOMPLETE_DATA_FILE_PATH + f'quickbits_preblock_headers.json')

    model.train()
    model.to(get_device())
    
    print(f"\tMinibatch Carts: {-(-len(quickbits)//BATCH_SAMPLES)}.")

    for i, mini_batch in enumerate(range(0, len(quickbits), BATCH_SAMPLES)):
        selected_quickbits = quickbits[mini_batch:mini_batch+BATCH_SAMPLES]
        

        dataloader = LoadDataLoaderWithData(selected_quickbits, checkpoint, hash_lookup=hash_lookup)

        
        for previous_block, current_block in dataloader:
            loss = 0
            batch, word_len = current_block.shape
      
            pred = model(previous_block.unsqueeze(1), current_block[:,:-4].unsqueeze(1))

            loss = loss_fn(pred[:,:4,:].reshape(-1, tokenizer.vocab_size), current_block[:, -4:].reshape(-1)) + ZerosFromPredictions(pred, current_block)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            total_loss += loss.detach().item()

            opt.step()
            sched.step()

            total_items += batch
        
        total_batches += 1

        loss_per_item = total_loss/total_items
        checkpoint.update_quickbit_count(count=batch)

        saved_emoji = "🗑️"
        if (loss_per_item < checkpoint.min_loss) | (i == (-(-len(quickbits)//BATCH_SAMPLES))):
            if (loss_per_item < checkpoint.min_loss): checkpoint.min_loss = loss_per_item

            if checkpoint.train:
                quicksave_model(checkpoint, model)
                checkpoint = save_checkpoint(checkpoint, quiet=True)
                saved_emoji = "🌠"


        if (i % 1 == 0):
            epoch_time_str, next_epoch_done_at_str = time_since_start(start_time, i)
            suffix = f'T.Loss/Item[{loss_per_item:.8f}]  T.Loss/Batch[{total_loss/total_batches:.8f}]  ⏳[{epoch_time_str}] 🕓[{next_epoch_done_at_str}]  {saved_emoji}'
            progress(i+1, -(-len(quickbits)//BATCH_SAMPLES), prefix=f"[🛤️ ]", suffix=suffix)
            

    return total_loss / total_items

#----------------------------------------------------------------
#----------------------------------------------------------------
def StationHub(model, opt, sched, loss_fn, dataloader, checkpoint):
    """ Distribute epoch cycles over the Train, Validate, and Evaluate functions & collect loss stats. """
    # Used for plotting later on
    train_loss, validation_loss = 0, 0
    train_loss_list, validation_loss_list, closeness_list = [], [], []
    train_time = {}
    train_time_list = []
    start_time = datetime.now()
    model.to(get_device())

    print(f"[🚦]  Starting Train Loop .... [1/{checkpoint.epoch}] @[{str(start_time).split('.', 2)[0]}]")

    for current_epoch in range(checkpoint.epoch):
        start_time = datetime.now()
        if checkpoint.train:
            #random.shuffle(dataloader.train)
            train_loss = TrainStation(model, opt, sched, loss_fn, dataloader.train, checkpoint)
            train_loss_list += [train_loss]
        t_time = datetime.now()
        train_time.update({"train_time": str(t_time - start_time).split('.', 2)[0]})

        if (current_epoch == 0) & (get_device() == 'cuda'):
            gpu_memory_used = torch.cuda.max_memory_allocated()
            checkpoint.gpu_usage.update({"gpu_memory_used": gpu_memory_used})


        if checkpoint.validate:
            validation_loss = ValidationStation(model, loss_fn, dataloader.validate, checkpoint)
            validation_loss_list += [validation_loss]
        v_time = datetime.now()
        train_time.update({"validation_time": str(v_time - t_time).split('.', 2)[0]})

        ShowTrainTime(current_epoch, checkpoint.epoch, start_time, (train_loss, validation_loss))


        if checkpoint.deploy:
            closeness_list += [EvaluationStation(model, dataloader.test, checkpoint)]

        d_time = datetime.now()
        train_time.update({"deploy_time": str(d_time- v_time).split('.', 2)[0]})

        train_time_list.append([train_time])

        checkpoint.losses.update({checkpoint.retrain: {'t_loss': train_loss_list,
                                                       'v_loss': validation_loss_list,
                                                       'closeness': closeness_list,
                                                       'checkpoint_times': train_time_list}})
        checkpoint = save_checkpoint(checkpoint)


    del opt, sched, loss_fn, dataloader, checkpoint, start_time
    return train_loss_list, validation_loss_list, closeness_list, train_time_list, model






#----------------------------------------------------------------
#----------------------------------------------------------------
def RunModelTrain( dataloader_cart, checkpoint  ):
    print(f"[🚂]  Chooo Chooo! All Aboard the Model Train ....")
    print(f"\tQuickBits Imported: ({len(dataloader_cart.train)}) -> On Train: ({BATCH_SAMPLES})")

    # Set Hyperparameters
    num_epochs      = checkpoint.epoch
    epoch_print_num = num_epochs // 8
    if num_epochs < 8 : epoch_print_num = num_epochs // num_epochs
    learning_rate   = checkpoint.lr
    print(f"\tLearning Rate: {learning_rate} -> ({round(learning_rate * MAX_INT)} as Nonce).")




    # Set CUDA/MPS Parameters
    print(f"\t{torch.backends.cudnn.is_available() = }")
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        device_properties = torch.cuda.get_device_properties(device)
        total_gpu_memory = device_properties.total_memory
        checkpoint.gpu_usage.update({"total_gpu_memory": total_gpu_memory})
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    print(f"\t{torch.backends.mps.is_available()   = }")
    if torch.backends.mps.is_available():
        torch.backends.mps.enabled = True
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_type = checkpoint.label

        
    if "📏" in model_type :
        config = ModelConfig()
        checkpoint.config_mdl = asdict(config)
        model = Torch_Transformer( **asdict(config) ).to(device)
        save_json_file({'model':str(model)}, f'checkpoints/{checkpoint.checkpoint_file()}_(model_def).json')
        optimizer = torch.optim.SGD(model.parameters(), lr=checkpoint.lr, momentum=0.95, weight_decay=0.00012345 )
        loss_fn = nn.CrossEntropyLoss( )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(dataloader_cart.train), epochs=checkpoint.epoch)

  
    for params in model.parameters():
        if params.dim() > 1:
            nn.init.xavier_normal_(params)
            #nn.init.normal_(params)

    if checkpoint.load_model() :
        model_path_name = f'{checkpoint.checkpoint_file()}'
        model           = LoadSavedModel(model_path_name, model)
        loaded_checkpoint = LoadSavedCheckpoint(model_path_name)
        if loaded_checkpoint != None: checkpoint = loaded_checkpoint
        print(f"\t{checkpoint = }")
    checkpoint.update_checkpoint()



    train_loss_list, validation_loss_list, closeness_list, train_time_list, model = StationHub(model, optimizer, scheduler, loss_fn, dataloader_cart, checkpoint)


    checkpoint.losses.update({checkpoint.retrain: {'t_loss':train_loss_list, 'v_loss':validation_loss_list, 'closeness':closeness_list, 'checkpoint_times': train_time_list}})


    checkpoint = save_checkpoint(checkpoint)
    quicksave_model(checkpoint, model)

    EndMessage(checkpoint)

    del model, checkpoint, dataloader_cart, optimizer, scheduler, loss_fn
    return



#----------------------------------------------------------------
def EndMessage(checkpoint):
    print(f"FINAL Checkpoint Name: {checkpoint.checkpoint_name()}  ReTrained: {checkpoint.retrain}   Total QuickBits: {checkpoint.qb}")
    print(f"[✔] ....  Done! [⭐️]\n[⭐️]✨" + f"✨"*90 + "✨[⭐️]\n")
    return


#----------------------------------------------------------------
def quicksave_model(checkpoint, model):
    file_name_path_model_state_dict = CHECKPOINT_FILE_PATH + f"QuickBitsNN_Checkpoint_({checkpoint.checkpoint_name()}).pth"
    torch.save(model.state_dict(), file_name_path_model_state_dict )
    file_name_path_model = CHECKPOINT_FILE_PATH + f"QuickBitsNN_Checkpoint_({checkpoint.checkpoint_name()})model.pt"
    torch.save(model, file_name_path_model )
    file_name_path_checkpoint_dict = CHECKPOINT_FILE_PATH + f"QuickBitsNN_Checkpoint_({checkpoint.checkpoint_name()}).json"
    SaveCheckpoint(checkpoint, file_name_path_checkpoint_dict)
    return

#----------------------------------------------------------------
def SaveModelTrained(checkpoint, model):
    print(f"[🚉 ]  Saving Model Checkpoint as State Dictionary ....")
    file_name_path_model_state_dict = CHECKPOINT_FILE_PATH + f"QuickBitsNN_Checkpoint_({checkpoint.checkpoint_name()}).pth"
    torch.save(model.state_dict(), file_name_path_model_state_dict )
    print(f"[✔︎] .... Model State Saved as: \"{file_name_path_model_state_dict}\"\n")
    return


#----------------------------------------------------------------
def save_checkpoint(checkpoint, quiet=False):
    if not quiet: print(f"[🚉 ]  Saving Checkpoint as Dictionary ....")
    file_name_path_checkpoint_dict = CHECKPOINT_FILE_PATH + f"QuickBitsNN_Checkpoint_({checkpoint.checkpoint_name()}).json"
    SaveCheckpoint(checkpoint, file_name_path_checkpoint_dict)
    if not quiet: print(f"[✔︎] .... Checkpoint Saved as: \"{file_name_path_checkpoint_dict}\"\n")


    dataloader_cart = []
    model = []

    return checkpoint


#----------------------------------------------------------------
def SaveCheckpoint(data, filename):
    serialized_data = data
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__, ensure_ascii=False)


#----------------------------------------------------------------
def LoadCheckpoint(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        data = CheckPoint(**data)
        return data


#----------------------------------------------------------------
def save_trained_model( model ):
    torch.save(model.state_dict(), MODEL_FILE_PATH / f"quickbits_{datetime.now()}.pth" )


#----------------------------------------------------------------
#----------------------------------------------------------------


