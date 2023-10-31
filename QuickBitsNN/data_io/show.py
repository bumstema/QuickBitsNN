import sys, os, os.path
import pandas as pd
import numpy as np
import random
import torch
import json
import math
import cv2
import PIL
from PIL import Image
from typing import List
from datetime import datetime, timedelta, time, date

import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints
import torchvision.transforms as transforms




from ..framework.quickbit import QuickBit, load_quickbits
from ..framework.constants import BATCH_SAMPLES, MAX_INT
from ..framework.tokenizer import Tokenizer
from ..framework.functions import HeaderNonceMiningHash, LeadingZeros, HexValueAsPercent
from ..framework.functions import little_endian_hex_to_int, big_endian, LeadingZeros, BitsZeros
from ..data_io.const import  DATA_FILE_PATH, INCOMPLETE_DATA_FILE_PATH, IMAGES_FILE_PATH
from ..data_io.sort import AppendQuickBitsFile
from ..data_io.utils import load_json_file, save_json_file
from ..environment.state import State


#----------------------------------------------------------------
#----------------------------------------------------------------

def ShowLastQuickBit( filename ):
    print(f"Importing QuickBits from File: {filename}")
    quickbits  =  load_quickbits(INCOMPLETE_DATA_FILE_PATH + f'{filename}')
    print(f"Total: {len(quickbits)}")
    print(f"QuickBit data... ")
    qb =  quickbits[-1]
    print(f"\n\n{qb.ver = }\n{qb.prev_block = }\n{qb.mrkl_root = }\n{qb.time = }\n{qb.bits = }\n{qb.nonce = }")
    print(f"\n{qb.hash}")



#----------------------------------------------------------------
def AverageQuickBitTokens( quickbits ):
    tokenizer = Tokenizer()
    head = [ tokenizer.tokenize(qb.build_header(decode=True)) for qb in quickbits]
    print(f"{len(head) = }")
    head = torch.stack(head)
    print(f"{head.shape = }")
    average_tokens = torch.mean(head.to(dtype=torch.float64), dim=0, keepdim=False)
    print(f"{average_tokens.shape = }")
    long_head = average_tokens.to(dtype=torch.long)
    print( f"{long_head = }" )
    ave_preblock_header = tokenizer.detokenize( long_head )
    print( f"{ave_preblock_header = }")

    ave_ver = little_endian_hex_to_int(ave_preblock_header[0:8])
    ave_block_hash = big_endian(ave_preblock_header[8:8+64])
    ave_mrklroot = big_endian(ave_preblock_header[8+64:8+64+64])
    ave_time = little_endian_hex_to_int(ave_preblock_header[-24:-16])
    ave_bits = little_endian_hex_to_int(ave_preblock_header[-16:-8])
    ave_nonce = little_endian_hex_to_int(ave_preblock_header[-8:])
    
    print( f"{ave_ver = }")
    print( f"{ave_block_hash = }  {len(ave_block_hash)}  zeros: {LeadingZeros(ave_block_hash)}")
    print( f"{ave_mrklroot   = }  {len(ave_mrklroot)}")
    print( f"{ave_time = }  date: {datetime.fromtimestamp(ave_time).strftime('%c')}")
    print( f"{ave_bits = }  target zeros: {BitsZeros(ave_bits)}")
    print( f"{ave_nonce = }  midpoint: {ave_nonce/(2**31)}")

    std_tokens = torch.std(head.to(dtype=torch.float64), dim=0, keepdim=False)
    print(f"{std_tokens.shape = }")
    print(f"{std_tokens = }")

    """
    ave_ver = 270209025
    ave_block_hash = '00000000060c142832557e7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f'  64  zeros: 9
    ave_mrklroot   = '7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f'  64
    ave_time = 1451196287  date: Sun Dec 27 01:04:47 2015
    ave_bits = 403931781  target zeros: 16
    ave_nonce = 1954446448  midpoint: 0.9101100489497185
    """
    data = {'mean_tokens': average_tokens.tolist(), 'std_tokens':std_tokens.tolist()}
    save_json_file(data, os.getcwd() + '/data/Average_QuickBit_Tokens.json')
    exit()
    return

#----------------------------------------------------------------
def BitBins( quickbits ):
    tokenizer = Tokenizer()
    head_bits = [ tokenizer.tokenize(qb.build_header_bits(decode=True)) for qb in quickbits]
    print(f"{head_bits.shape = }")  # [N,  160]
    head_bits = torch.stack(head_bits)
    print(f"{head_bits.shape = }")
    bits_hist = torch.histogramdd(head_bits, 1)
    print(f"{bits_hist = }")
    print(f"{bits_hist.shape = }")
    exit()
    return



#----------------------------------------------------------------
def MeanState():

    data = load_json_file( os.getcwd() + '/data/Average_QuickBit_Tokens.json')
    PREBLOCK_TOKEN_AVERAGES = data['mean_tokens']
    PREBLOCK_TOKEN_STD = data['std_tokens']
    
    mean_state = State(block=PREBLOCK_TOKEN_AVERAGES.to(dtype=torch.long) )

    std_range = np.arange(-3,3.05,0.05)
    tokens_around_mean = [ (PREBLOCK_TOKEN_AVERAGES + (std_*PREBLOCK_TOKEN_STD) ).to(dtype=torch.long)  for std_ in std_range ]
    tokens_around_mean = [torch.remainder( tokens_, 255) for tokens_ in tokens_around_mean]

    std_states = [ State(block=tokens_)  for tokens_ in tokens_around_mean ]
    state_pixels = [state_.render() for state_ in std_states]
    full_pixels = np.array(state_pixels).squeeze(1)
    
    print(f"{full_pixels = }")
    print(f"{full_pixels.shape = }")

    BasicPixelPlot(full_pixels, filename=f'preblock_mean_and_std_pixels.png')
    return


#----------------------------------------------------------------
def BasicPixelPlot( data , filename=f'pixels.png'):
    pixel_plot = plt.figure(dpi=600)
    fig, ax = plt.subplots(1,1, figsize=(8, 6))
    

    ax.set_xticks(np.array((1,9,17,18,19,20)), labels=('Version','Prev.Block Hash','Merkle Root','Time','Bits','Nonce'))
    plt.xticks(rotation=-45, ha="left", va="top", rotation_mode="anchor")

    ax.set_yticks(np.arange(-15,20,5), labels=('-3σ','-2σ', '-σ', '<X>', 'σ', '2σ','3σ'))
    plt.rcParams.update({'font.size': 10})

    plt.xlabel('Parameter Pixels')
    #plt.ylabel('y axis')
    pixel_plot = plt.imshow( data, interpolation='none', origin='lower', extent=[0, 20, -15, 15])

    #plt.colorbar(pixel_plot)#, shrink=0.25)
    ax1_divider = make_axes_locatable(ax)
    # Add an Axes to the right of the main Axes.
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(pixel_plot, cax=cax1)
    
    #plt.tight_layout()
    plt.savefig( IMAGES_FILE_PATH / f'{filename}')
    plt.close()
    return

#----------------------------------------------------------------
def PlotQuickNBitsTimes(quickbits):
    # Create a dictionary with the time values
    data = {'time': [qb.time for qb in quickbits]}
    
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    
    # Plot the time values
    plt.figure(figsize=(10, 6))
    plt.plot(df['time'])
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.title('Quickbits Time')
    
    # Calculate and plot the standard deviation
    std_dev = df['time'].std()
    plt.axhline(std_dev, color='r', linestyle='--', label='Standard Deviation')
    plt.legend()
    
    # Show the plot
    plt.show()
    

#----------------------------------------------------------------
def GenerateNewZeroQuickBits( quickbits ):
    device      = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bins = {}
    quickbits_saved = quickbits
    trials = 20000000
    try:
        for i in range(trials):

            random_quickbits =  random.sample(quickbits_saved, k=BATCH_SAMPLES)
            random_nonce = random.randint(0, 2**32 - 1)

            qb_calc_hash = [ (random_quickbits[n],  random_quickbits[n].leading_zeros_of_hash(int(random_nonce))) for n in range(BATCH_SAMPLES)]

            for qb, hash_zeros in qb_calc_hash:
                if hash_zeros not in bins:
                    bins[hash_zeros] = []
                if hash_zeros == 0 :
                    if random.random() >= 0.00005 :
                        continue

                if hash_zeros == 1 :
                    if random.random() >= 0.0005 :
                        continue

                if hash_zeros == 2 :
                    if random.random() >= 0.007 :
                        continue

                if hash_zeros == 3 :
                    if random.random() >= 0.1 :
                        continue

                qb.nonce = random_nonce
                qb.hash = qb.calculate_mining_hash(qb.nonce)
                bins[hash_zeros].append(qb)

            print(f"Running: [{i: >7}/{trials}]")
            print(f"Total Bins: {len(bins)}.")
            [print(f"Leading Zeros: {bin: >10}  Items: {len(entries): >8}") for bin,entries in bins.items() ]


        for leading_zeros, entries in bins.items():
            filename = INCOMPLETE_DATA_FILE_PATH + f'generated_v2_quickbins_({leading_zeros})0s.json'
            #append_quickbits_file(entries, filename)
            AppendQuickBitsFile(entries, filename)

    except:
        print(f"Error: Dumping Files..")
        for leading_zeros, entries in bins.items():
            filename = INCOMPLETE_DATA_FILE_PATH + f'generated_v2_quickbins_({leading_zeros})0s.json'
            #append_quickbits_file(entries, filename)
            AppendQuickBitsFile(entries, filename)
            
            
            
#----------------------------------------------------------------
#----------------------------------------------------------------

# Plots each Preblock Header parameter as a single #RRGGBBAA pixel
# 1x  pixel: version, bits, time, nonce
# 8x pixels: previous_block, merkle_root
# ========
# 20 pixels total  np size = [w,h,c] -> [20,1,4] ==> torch shape = [c, w, h] -> [4,20,1]

#def QuickBitHeaderPixels():
    #fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6), subplot_kw={'xticks': [], 'yticks': []})
    #ax.imshow(grid, interpolation=interp_method, cmap='viridis')
    #return

#----------------------------------------------------------------
def header_colour(qb='version'):
    HEADER_COLOURS = {'version':"#B57EDC",
        '-log(previous block hash)':"#929591",
        'merkel root':"#2969B0",
        'bits':"#008000",
        'time':"#40E0D0",
        'nonce':"#DBB40C"  #"#FAC205"
        
}
    try:
        return HEADER_COLOURS.get(qb)
    except:
        return f"#F5F5DC"
    
#----------------------------------------------------------------
def PlotQuickBits(quickbits):
    df = pd.DataFrame({'nonce': [bit.nonce for bit in quickbits], 'time': [bit.time for bit in quickbits]})
    
    # Convert Unix time to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print(df['time'].min())
    # Call the functions with the generated data
    matplotlib_scatter(df)
    return

#----------------------------------------------------------------
def violinplot(df, df_v, name_x="", name_y="", title="", filename="temp.png"):
    sns.set(style="whitegrid")
    combined = pd.concat([df, df_v], ignore_index=True)

    f, ax = plt.subplots(dpi=600)
    sns.violinplot(data=combined, x=f"{name_x}", y=f"{name_y}", palette="Set2",  orient='h', bw=0.001, inner=None, scale="width", linewidth=0.75)
    
    f.suptitle(f'{title}', fontsize=18, fontweight='bold')
    ax.set_xlabel(f"{name_x}",size = 16,alpha=0.7)
    ax.set_ylabel(f"{name_y}",size = 16,alpha=0.7)
    plt.grid(True)
  
    plt.savefig( IMAGES_FILE_PATH / f'{filename}')
    plt.close()
    return


#----------------------------------------------------------------
def matplotlib_histogram(df, name_x="", filename="temp.png", num_bins=250 ):

    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    plt.xlabel(f'{name_x}', fontsize=18)
    plt.ylabel('Probability Density', fontsize=18)
    plt.grid(True)
    plt_colour = header_colour(qb=f'{name_x}')

    plt.legend()
    plt.tight_layout()
    
    bin_edges = np.linspace(np.min(df[f'{name_x}']), np.max(df[f'{name_x}']), num=num_bins)
    
    plt.hist( df[f'{name_x}'], bins = bin_edges, density=True, color=plt_colour, alpha=0.7, edgecolor='white', align='left' )

    plt.savefig( IMAGES_FILE_PATH / f'{filename}')
    plt.close()
    return

#----------------------------------------------------------------
def matplotlib_histogram_split_data(df, df_v, name_x="", title="", filename="temp.png", num_bins=250 ):

    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    bin_edges = np.linspace(np.min(df[f'{name_x}']), np.max(df[f'{name_x}']), num=num_bins)
    combined = pd.concat([df[f'{name_x}'], df_v[f'{name_x}']], ignore_index=True)
    
    plt_colour = header_colour(qb=f'{name_x}')
    print(f"{plt_colour=}")
    plt.hist( combined, bins = bin_edges, density=True, color=plt_colour, alpha=0.7, edgecolor='white', align='left' )
    

    plt.xlabel(f'{name_x}', fontsize=18)
    plt.ylabel('Probability Density', fontsize=18)
    plt.title(f'{title}', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig( IMAGES_FILE_PATH / f'{filename}')
    plt.close()
    return


#----------------------------------------------------------------
def matplotlib_scatter_split_data(df, df_v, name_x="", name_y="", title="", filename="temp.png"):
    
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    plt.title(f'{title}', fontsize=16)
    plt.xlabel(f'{name_x}', fontsize=18)
    plt.ylabel(f'{name_y}', fontsize=18)
    plt_colour = header_colour(qb=f'{name_y}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    
    # Convert from Unix Time to Year-Month-Day format
    if 'time' == name_x:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df_v['time'] = pd.to_datetime(df_v['time'], unit='s')
        plt.xticks(rotation=-45)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.axvline(x=df_v[f'{name_x}'][0], color="#FF0000", linestyle=':', label=None)

    
    plt.scatter(df[f'{name_x}'], df[f'{name_y}'], c=plt_colour, label='Training Data', s=0.00075)
   
    plt.scatter(df_v[f'{name_x}'], df_v[f'{name_y}'], c=f"#FF0000", label=f'Validation Data', s=0.00075)
    

    print( f'{filename}')
    plt.savefig( IMAGES_FILE_PATH / f'{filename}')
    plt.close()
    return

    
#----------------------------------------------------------------
def PlotProbabilityMapsNonceBitsTime(quickbits, validation_qb):
    df = pd.DataFrame({'nonce': [bit.nonce for bit in quickbits],'log(nonce)': [np.log(bit.nonce) for bit in quickbits], 'time': [bit.time for bit in quickbits], 'bits': [bit.bits for bit in quickbits], 'version': [bit.ver for bit in quickbits], 'zeros': [bit.hash_zeros() for bit in quickbits], 'merkel root': [HexValueAsPercent(bit.mrkl_root,64) for bit in quickbits], '-log(merkel root)': [-np.log(HexValueAsPercent(bit.mrkl_root,64)) for bit in quickbits],  '-log(previous block hash)': [-np.log(HexValueAsPercent(bit.prev_block,64)) for bit in quickbits], 'previous block hash': [bit.prev_block for bit in quickbits], 'hash': [bit.hash for bit in quickbits], 'unix time': [bit.time for bit in quickbits]})
    df['cal_time'] = pd.to_datetime(df['time'], unit='s')

    df_v = pd.DataFrame({'nonce': [bit.nonce for bit in validation_qb],'log(nonce)': [np.log(bit.nonce) for bit in validation_qb], 'time': [bit.time for bit in validation_qb], 'bits': [bit.bits for bit in validation_qb], 'version': [bit.ver for bit in validation_qb], 'zeros': [bit.hash_zeros() for bit in validation_qb], 'merkel root': [HexValueAsPercent(bit.mrkl_root,64) for bit in validation_qb], '-log(merkel root)': [-np.log(HexValueAsPercent(bit.mrkl_root,64)) for bit in validation_qb],'-log(previous block hash)': [-np.log(HexValueAsPercent(bit.prev_block,64)) for bit in validation_qb], 'previous block hash': [bit.prev_block for bit in validation_qb], 'hash': [bit.hash for bit in validation_qb], 'unix time': [bit.time for bit in validation_qb]})
    
    
    hash_to_nonce_lookup = {}
    for bit in quickbits:
        hash_to_nonce_lookup.update({bit.hash: bit.nonce})
    
    nonce_by_nonce=[]
    for bit in quickbits[1:]:  #sorted(  , key=lambda x : x.time):
        nonce_by_nonce.append([hash_to_nonce_lookup[bit.prev_block], bit.nonce])
        
    prev_nonce, nonce = list(map(list, zip(*nonce_by_nonce)))
    noncedf = pd.DataFrame({'nonce':nonce, 'prev_nonce':prev_nonce})
    
    
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    plt.legend()
    plt.grid(True)

    plt.xlabel(f'time', fontsize=18)
    plt.xticks(rotation=-45)
    
    plt.ylabel(f'Δ(nonce)', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt_colour = header_colour(qb='nonce')

    plt.title(f'Difference in Nonce between Sequential Blocks', fontsize=16)
    
    plt.scatter( pd.to_datetime(df['time'], unit='s')[1:], noncedf['nonce']-noncedf['prev_nonce'] , c=plt_colour, label='QuickBits', s=0.000075)

    plt.savefig( IMAGES_FILE_PATH / 'nonce_difference_vs_time.png')
    plt.close()
    
 
    matplotlib_scatter_split_data(df, df_v, 'time', 'zeros', 'Leading Zeros Over Time', 'leading_zeros_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'bits', 'Bits Value Over Time', 'bits_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'nonce', 'Nonce Value Over Time', 'nonce_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'log(nonce)', 'log(Nonce) Value Over Time', 'log_nonce_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'version', 'Block Chain Version Over Time', 'version_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'merkel root', 'Normalized Merkel Root Over Time', 'mrklroot_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', '-log(merkel root)', '-log(Normalized Merkel Root) Over Time', 'log_mrklroot_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', '-log(previous block hash)', '-log(Normalized Previous Block Hash) Over Time', 'prevhash_vs_time.png')
    matplotlib_scatter_split_data(df, df_v, 'time', 'unix time', 'Unix Time Over Time', 'unix_time_vs_time.png')


    matplotlib_scatter_split_data(df, df_v, 'nonce', 'merkel root', 'Merkle Root vs. Nonce', 'mrklroot_vs_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'nonce', '-log(merkel root)', '-log(Normalized Merkel Root) vs. Nonce', 'log_mrklroot_vs_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'nonce', '-log(previous block hash)', '-log(Normalized Previous Block Hash) vs. Nonce', 'log_prevhash_vs_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'nonce', 'version', 'Version vs. Nonce', 'version_vs_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'nonce', 'bits', 'Bits vs. Nonce', 'bits_vs_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'nonce', 'zeros', 'Zeros vs. Nonce', 'zeros_vs_nonce.png')
 
    matplotlib_scatter_split_data(df, df_v, 'zeros', '-log(previous block hash)', '-log(PrevHash) vs. Zeros ', 'log_prevhash_vs_zeros.png')

    matplotlib_scatter_split_data(df, df_v, 'bits', '-log(previous block hash)', 'Bit vs. PrevHash', 'log_prevhash_vs_bits.png')
    matplotlib_scatter_split_data(df, df_v, 'bits', 'zeros', 'Leading Zeros vs. Bits', 'leading_zeros_vs_bits.png')


    matplotlib_scatter_split_data(df, df_v, 'log(nonce)', '-log(merkel root)', '-log(Normalized Merkel Root) vs. log(Nonce)', 'log_mrklroot_vs_log_nonce.png')
    matplotlib_scatter_split_data(df, df_v, 'log(nonce)','-log(previous block hash)', '-log(Previous Block Hash) vs. log(Nonce)', 'log_prevhash_vs_log_nonce.png')


    matplotlib_scatter_split_data(df, df_v, '-log(previous block hash)', '-log(merkel root)', '-log(Normalized Previous Block Hash) vs. -log(Merkle Root)', 'log_prevhash_vs_log_mrklroot.png')
    

    matplotlib_histogram_split_data(df, df_v, 'zeros', 'Histogram of Leading Zeros', f'leading_zeros_histogram.png', num_bins=((16*2)+1))
    matplotlib_histogram_split_data(df, df_v, 'nonce', 'Histogram of Nonce Values', f'nonce_histogram.png', num_bins=128)
    matplotlib_histogram_split_data(df, df_v, 'log(nonce)', 'Histogram of log(Nonce) Values', f'log_nonce_histogram.png', num_bins=128)
    matplotlib_histogram_split_data(df, df_v, 'bits', 'Histogram of Bits Values', f'bits_histogram.png', num_bins=64)
    matplotlib_histogram_split_data(df, df_v, 'version', 'Histogram of Nomalized Block Chain Version', f'version_histogram.png', num_bins=64)
    matplotlib_histogram_split_data(df, df_v, 'merkel root', 'Histogram of Normalized Merkel Root', f'mrklroot_histogram.png', num_bins=256)
    matplotlib_histogram_split_data(df, df_v, '-log(previous block hash)', 'Histogram of -log(Normalized Previous Block Hash)', f'prevhash_histogram.png', num_bins=64)
    matplotlib_histogram_split_data(df, df_v, 'unix time', 'Histogram of Unix Time', f'unix_time_histogram.png', num_bins=128)

    

    violinplot(df, df_v,  'nonce', 'zeros','Probability of Nonce per Leading Zeros', 'violinplot_zeros_vs_nonce.png')



    hash_time = {}
    for bit in quickbits:
        hash_time.update({bit.hash: bit.time})
    for bit in validation_qb:
        hash_time.update({bit.hash: bit.time})
    
   
    dt = []
    for bit in quickbits[1:]:
        dt += [{'time': bit.time, 'Δ time': bit.time - hash_time[bit.prev_block]}]

    vb_dt = []
    for bit in validation_qb:
        vb_dt += [{'time': bit.time, 'Δ time': bit.time - hash_time[bit.prev_block]}]


    dt_df = pd.DataFrame(dt)
    vb_dt_df= pd.DataFrame(vb_dt)
    

    matplotlib_scatter_split_data(dt_df, vb_dt_df, 'time', 'Δ time', 'Block Generation Rate Over Time', 'block_generation_rate_vs_time(1).png')
    matplotlib_histogram_split_data(dt_df, vb_dt_df, 'Δ time', 'Histogram of Block Generation Rate', f'block_generation_rate_histogram(1 ).png', num_bins=64)


    print(f"Plot Complete!")
    exit()
    return
    
    #----------------------------////////////////
    """

    df_t = df[['previous block hash','hash','time']]
    # Merge the DataFrame with itself based on 'prev_block' and 'block'
    merged_df = df_t.merge(df_t, left_on='hash', right_on='previous block hash', suffixes=('', '_block'))

    # Calculate the time difference between 'time_block' and 'time_prev'
    merged_df['Δ time'] = merged_df['time_block'] - merged_df['time'] - timedelta(seconds=600)
    merged_df['Δ time'] = merged_df['Δ time'].astype(int)
    # Drop unnecessary columns and keep the relevant ones
    result_df = merged_df[['time',  'time_block', 'Δ time']]


    df_t_v = df_v[['previous block hash','hash','time']]
    # Merge the DataFrame with itself based on 'prev_block' and 'block'
    merged_df_v = df_t_v.merge(df_t_v, left_on='hash', right_on='previous block hash', suffixes=('', '_block'))

    # Calculate the time difference between 'time_block' and 'time_prev'
    merged_df_v['Δ time'] = merged_df_v['time_block'] - merged_df_v['time'] - timedelta(seconds=600)
    merged_df_v['Δ time'] = merged_df_v['Δ time'].astype(int)
    # Drop unnecessary columns and keep the relevant ones
    result_df_v = merged_df_v[[ 'time', 'time_block', 'Δ time']]



    matplotlib_scatter_split_data(result_df, result_df_v, 'time', 'Δ time', 'Block Generation Rate Over Time', 'block_generation_rate_vs_time.png')
    matplotlib_histogram_split_data(result_df, result_df_v,  'Δ time', 'Histogram of Block Generation Rate', f'block_generation_rate_histogram.png', num_bins=64)
    """

    
    # Create a new figure
    plt.figure()

    plt.scatter(df['time'], df['nonce'] , 'Time', 'Nonce', c='b', label='Training Data')
    plt.scatter(df_v['time'], df_v['zeros'] , 'Time', 'Nonce', c='r', label='Validation Data')
    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Nonce')
    plt.title('Nonce Values Over Time')
    # Add legend
    plt.legend()
    plt.savefig( IMAGES_FILE_PATH / 'nonce_vs_time.png')




    # Create a new figure
    plt.figure()

    plt.scatter(df['time'], df['nonce'] , 'Time', 'Nonce', c='b', label='Training Data')
    plt.scatter(df_v['time'], df_v['zeros'] , 'Time', 'Nonce', c='r', label='Validation Data')
    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Nonce')
    plt.title('Nonce Values Over Time')
    # Add legend
    plt.legend()
    plt.savefig( IMAGES_FILE_PATH / 'nonce_vs_time.png')




    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with three subplots
    plot = []
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # Plot version vs time
    ax1.scatter(df['time'], df['version'] , 'Time', 'Version')

    
    # Plot bits vs time
    ax2.scatter(df['time'], df['bits'], 'Time', 'Bits')

    

    ax3.scatter(df['time'], df['zeros'] , 'Time', 'Leading Zeros')



    fig, ax = plt.subplots()
    num_bins = 200
    # shift the dates up and into a new column
    #df['time'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    df['dates_shift'] = df['time'].shift(-1)
    df['time_diff'] = (df['dates_shift'] - df['time']).abs() #/ pd.Timedelta(seconds=1)
    del df['dates_shift']
    
    #n, bins, patches = ax.hist( df['time_diff'], num_bins, density=True)
    #ax.set_xlabel('Time For Solving Next Block')
    #ax.set_ylabel('Probability Density')
    #fig.tight_layout()
    #plt.show()
 
    
    ax4.scatter(df['time'], df['time_diff'], 'Time', 'Delta Time Difference')

    for ax in fig.get_axes():
        ax.label_outer()

    plt.show()



    fig, ax = plt.subplots()
    # the histogram of the data
    num_bins = 250
    # example data
    mu =  df['nonce'].mean()  # mean of distribution
    sigma =  df['nonce'].std()  # standard deviation of distribution
    x = mu + sigma * np.random.randn(437/2)

    n, bins, patches = ax.hist( df['nonce'], num_bins, density=True)
    y = ((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    ax.plot(bins, y, '--')
    ax.set_xlabel('Nonce')
    ax.set_ylabel('Probability Density')
    fig.tight_layout()
    plt.show()

    df['time'] = pd.to_datetime(df['time'], unit='s')



    #plt.savefig('sample_plot.png')

#----------------------------------------------------------------
def PlotGraph( param_dict, filename='graph.png' ):

    df = pd.DataFrame(param_dict)
    x = df['episode']
    reward = df['total_reward']
    goal = df['reward']
    
     
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    #ax = plt.subplot(111)
    plt.scatter(x, (reward-goal), s=1.75)

    majorLocator   = MultipleLocator(1)

    plt.xlabel(f'Episode (#)', fontsize=18)
    plt.ylabel(f'(T.Rewards - Goal)', fontsize=18)
    plt.title(f'level-1', fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig( f'{filename}')
    plt.close()
    return

#----------------------------------------------------------------
def ProbabilityDensityHistogram(param_dict, name_x='', title='', filename='temp.png', num_bins=128 ):

    df = pd.DataFrame(param_dict)
    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    
    #plt.legend()
    plt.tight_layout()
    plt.title(f'{title}', fontsize=16)
    plt.xlabel(f'{name_x}', fontsize=18)
    plt.ylabel('Probability Density', fontsize=18)
    #plt.grid(True)
    #plt_colour = header_colour(qb=f'{name_x}')


    
    bin_edges = np.linspace(np.min(df[f'{name_x}']), np.max(df[f'{name_x}']), num=num_bins)
    
    plt.hist( df[f'{name_x}'], bins = bin_edges, density=True, alpha=0.7, edgecolor='white', align='left' )

    plt.savefig( f'{filename}')
    plt.close()
    return



#------------------------------------------------------------
def PlanarDensityHistogram(param_dict, name_x='', name_y='', title='',  filename='graph.png' ):

    plt.figure(dpi=600)
    plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    plt.title(f'{title}', fontsize=16)
    plt.xlabel(f'{name_x}', fontsize=18)
    plt.ylabel(f'{name_y}', fontsize=18)
    plt.grid(True)
 

    df = pd.DataFrame(param_dict)
    x = df[f'{name_x}']
    y = df[f'{name_y}']

    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(x, y,  bins=64)
    ax.set_xlabel(f'{name_x}', fontsize=18)
    ax.set_ylabel(f'{name_y}', fontsize=18)

    plt.savefig( f'{filename}')
    plt.close()
    return

#----------------------------------------------------------------
def PlotPixels(images):
    colour_map = mpl.colormaps['gnuplot']
    num_images = len(images)
    num_rows = int(num_images ** 0.5)
    num_cols = (num_images + num_rows - 1) // num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8,8))
    axes = axes.flatten()
    for i, image in enumerate(images):
        ax = axes[i]
        ax.imshow(image, interpolation='nearest', origin='lower', cmap='Greens',)
        ax.axis('off')
        plt.tight_layout()
    plt.tight_layout()
    plt.show()


#----------------------------------------------------------------
def PlotEncodedEmbeddings(embeddings):
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(embeddings[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Channel {i+1}')
    plt.tight_layout()
    plt.show()




#----------------------------------------------------------------
#----------------------------------------------------------------
def process_raw_frame(np_rgb):
    pic = datapoints.Image(PIL.Image.fromarray(np_rgb.astype(np.uint8)))
    return(pic)

#----------------------------------------------------------------
def rgb2gray(rgb):
    return cv2.cvtColor( rgb, cv2.COLOR_RGB2GRAY)
    
#----------------------------------------------------------------
def np_grayscale(rgb_img):
    return np.dot(rgb_img[...,:3], [0.2989, 0.5870, 0.1140])
  
#----------------------------------------------------------------
def image_to_tensor(np_gray_img):
    pic = datapoints.Image(PIL.Image.fromarray((np_gray_img).astype(np.uint8)))
    pic = pic.type(dtype=torch.float32)
    pic = pic/255
    return pic
    
#----------------------------------------------------------------
def tensor_to_pil_image( pic ):
    return transforms.ToPILImage()(pic)
    
#----------------------------------------------------------------
def save_gif_frames(gif_frames, ms_per_frame=16.67, file_name=f'file_name.gif'):
    #print(f"Now saving gif...")
    gif_frames[0].save(f'{file_name}', format='GIF', append_images=gif_frames[1:], save_all=True, loop=0, duration= ms_per_frame, disposal=0, optimize=True)
    

