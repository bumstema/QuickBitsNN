import json, codecs
import sys, os, os.path
import numpy as np
import torch
import math
import gc

#------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


#------------------------------------------------------------
def save_json_file(data, filename):
    serialized_data = data
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__, ensure_ascii=False)

#------------------------------------------------------------
def load_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        data = json.load(f)
        return data

#------------------------------------------------------------
def unpack_nested_list(listed_lists):
    return [val for sublist in listed_lists for val in sublist]

#------------------------------------------------------------
def progress(count, total, prefix='', suffix=''):
    bar_len = 5
    filled_len = int(math.ceil(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '_' * (2*bar_len - 2*filled_len - 1) + '🚂' + '🚃' * filled_len

    sys.stdout.write(f'%s [%s] %s%s | {count:>5}/%s ... %s\r' % (prefix, bar, percents, '%',  total, suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben
    #if count == total: print(f"\n")




#----------------------------------------------------------------
def set_device_params():

    # Set CUDA/MPS Parameters
    print(f"\t{torch.backends.cudnn.is_available() = }")
    if torch.backends.cudnn.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.backends.cudnn.enabled = True
        device_properties = torch.cuda.get_device_properties(get_device())
        print(torch.cuda.get_device_properties(0))
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True


        gpu_memory_used = torch.cuda.max_memory_allocated()
        total_gpu_memory = device_properties.total_memory

        print(f"GPU Max Memory Allocated / Used : {gpu_memory_used// 1e9} GB / {total_gpu_memory// 1e9} GB")
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            print(f"Cuda Model using - BF16.")
            return 'bf16-true'
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
            print(f"Cuda Model using - F16.")
            return '16-true' #'16-mixed'
        
    print(f"\t{torch.backends.mps.is_available()   = }")
    if torch.backends.mps.is_available():
        torch.backends.mps.enabled = True
        gpu_memory_used = torch.mps.driver_allocated_memory()
        print(f"GPU Max Memory Allocated: {gpu_memory_used // 1e9} GB")
        return '32-true'

#----------------------------------------------------------------
def lexsort(keys, dim=-1):
    """
    https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/4
    """
    if keys.ndim < 2:
        raise ValueError(f"keys must be at least 2 dimensional, but {keys.ndim=}.")
    if len(keys) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(keys)=}.")
    
    idx = keys[0].argsort(dim=dim, stable=True)
    for k in keys[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx

#----------------
def dezeroize(x, full=True):
    if full:
        x[torch.logical_and(x >= 0, x <= 1e-8)] = 1e-8
        x[torch.logical_and(x < 0, x >= -1e-8)] = -1e-8
        return x
    # to handle 0 <= x <= 1e-4
    x[torch.logical_and(x >= 0, x <= 1e-4)] = 1e-4
    # to handle -1e-4 <= x < 0
    x[torch.logical_and(x < 0, x >= -1e-4)] = -1e-4
    return x
