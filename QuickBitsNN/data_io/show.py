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
import matplotlib
import plotly.graph_objects as go
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import matplotlib.dates as mdates
from matplotlib import colors, cm
from matplotlib.patches import StepPatch

import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import scipy.special


import torch.nn.functional as F
import gc

import seaborn as sns

from ..framework.quickbit import QuickBit, load_quickbits
from ..framework.constants import BATCH_SAMPLES, MAX_INT
from ..framework.tokenizer import Tokenizer
#from ..framework.tokenizer_single_digit import Tokenizer
from ..framework.functions import HeaderNonceMiningHash, LeadingZeros, HexValueAsPercent
from ..framework.functions import little_endian_hex_to_int, big_endian, LeadingZeros, BitsZeros
from ..framework.functions import LogitRankingsFromKnownSolutions
from ..data_io.const import  DATA_FILE_PATH, INCOMPLETE_DATA_FILE_PATH, IMAGES_FILE_PATH
from ..data_io.sort import AppendQuickBitsFile, find_files_with_similar_names
from ..data_io.utils import load_json_file, save_json_file, progress, dezeroize


#----------------------------------------------------------------
#----------------------------------------------------------------

#----------------------------------------------------------------
def pixelize(unbatched_2d_tensor: torch.Tensor, colourize=True, min_max=None) -> PIL.Image:

    pixels_ = unbatched_2d_tensor.clone().detach()
    
    if not colourize:
        a_max = torch.max(torch.abs(pixels_))
        pixels_ = 255 * ((pixels_/ (2*a_max)) + 0.5)
        pixels_ = pixels_.to(dtype=torch.long).cpu().numpy()
        pic = PIL.Image.fromarray((pixels_.astype(np.uint8)))
        return pic
    
    a, b = pixels_.shape
    ratio = a/b
    fig, axes = plt.subplots( 1, 1, figsize=(5,ratio*5), clear=True,  dpi=100)
    colour_map = matplotlib.colormaps['terrain']
    plt.rcParams.update({'font.size': 0})
    #pixels_ = dezeroize(pixels_, full=True)

    pixels_ = pixels_.cpu().numpy()
    #axes.pcolor(pixels_, norm=colors.LogNorm(vmin=np.min(pixels_), vmax=np.max(pixels_)), cmap=colour_map)
    if min_max is None:
        axes.pcolor(pixels_, cmap=colour_map)
    else:
        axes.pcolor(pixels_, norm=colors.Normalize(vmin=min_max[0], vmax=min_max[1]), cmap=colour_map)
    plt.axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    pic = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.rcParams.update({'font.size': 9})
    plt.clf()
    plt.axis('on')
    plt.close('all')
    gc.collect()
    return pic


#----------------------------------------------------------------
def PlotLogitemRankings(filepath=IMAGES_FILE_PATH, filename=f'', return_figure=True, rankings=None, logits=None, labels=None):
    if rankings is None:
        solved_digit_rankings = LogitRankingsFromKnownSolutions(logits.to(dtype=torch.float32), labels)
        ranks = np.array(solved_digit_rankings)
    else:
        ranks = np.array(rankings)
    n_bins = 256
    plt.rcdefaults()
    plt.figure(figsize=(16,11),  clear=True)
    plt.rcParams.update({'font.size': 9})
    plt.ylabel("Probability Density", fontsize=12)
    #plt.tight_layout()
    label_names  = ['Nonce[0]', 'Nonce[1]', 'Nonce[2]', 'Nonce[3]']
    colour_labels= ['tab:blue','tab:orange','tab:green','tab:red']

    #plt.title("Rankings of Blockchain Nonce Tokens within Validation Logits", fontsize=16)

    #plt.hist(ranks.T, n_bins, histtype='bar', density=True, label=label_names)

    #rank_prob_per_logit = [np.histogram(np.array(logit_ranks_), bins=256, density=True)[0] for logit_ranks_ in solved_digit_rankings]
    #rank_prob_per_logit = np.array(rank_prob_per_logit)


    num_rows, num_cols = 5, 1
    fig, axes = plt.subplots( num_rows, num_cols, figsize=(16, 11), clear=True, dpi=100)
    plt.grid(True)
    for i_, ranks_ in enumerate(ranks.T):
        sns.histplot(data=ranks_,
                     bins=n_bins,
                     kde=True,
                     stat='density',
                     label=label_names[i_],
                     ax=axes[i_],
                     element='step',
                     color=colour_labels[i_])
        axes[i_].axhline(y=(1/256.), color="000000", linestyle='-', linewidth=1., label=f'Random Chance', antialiased=True, snap=True)
        axes[i_].set_xlabel(f'', fontsize=10)
        axes[i_].set_ylabel(f'', fontsize=10)
        axes[i_].grid(True)
        axes[i_].legend(fontsize=8)
        axes[i_].set_xticks(np.arange(0, 256, 16))

        #(tan_x, tan_y) = curve_fit_ranks_to_negtan(rank_prob_per_logit[i_])
        #axes[i_].plot(tan_x, tan_y, linewidth=1.15)

    for i_, ranks_ in enumerate(ranks.T):
        sns.kdeplot(data=ranks_,
                     label=label_names[i_],
                     ax=axes[-1],
                     color=colour_labels[i_],
                    clip=[0,256])
        axes[i_].axhline(y=(1/256.), color="000000", linestyle='-', linewidth=1., label=f'Random Chance', antialiased=True, snap=True)
        axes[-1].set_xlabel(f'', fontsize=10)
        axes[-1].set_ylabel(f'', fontsize=10)
        axes[-1].grid(True)
        axes[-1].legend(fontsize=8)
        axes[-1].set_xticks(np.arange(0, 256, 16))

    plt.xlabel("Ranking below Top Logit (Number)", fontsize=12)
    fig.tight_layout()

    folder= f'logitem_ranks/'
    try:
        os.mkdir(filepath + folder)
    except:
        pass
    plt.savefig(filepath + folder + filename, dpi=150)
    gc.collect()
    if return_figure:
        return fig


    plt.clf()
    plt.close('all')
    gc.collect()
    return

#----------------------------------------------------------------

def max_token_select( nonce_logit, soft_max=True):
    digits, logits = nonce_logit.shape
    if soft_max: nonce_logit = F.softmax(nonce_logit, dim=-1)
    logit_index = torch.argmax(nonce_logit.cpu(), dim=-1)
    logit_index = torch.stack([logit_index,(torch.arange(digits)) ])

    return logit_index.tolist()


#----------------------------------------------------------------
def PlotLogitClassProbability( logits,
                               solved=None,
                               predicted=None,
                               filepath=IMAGES_FILE_PATH,
                               filename=f'',
                               return_figure=False,
                               prenormalized=True):
    # NOTE: (batch_size, vocab_dim, n_logits = logits.shape)
    logits = torch.clone(logits.detach())

    tknzr = Tokenizer()
    
    if not prenormalized:
        logits = F.softmax(logits, dim=1, dtype=torch.float32)

    logits = logits.permute(0, 2, 1)
    #logits = dezeroize(logits, full=True)

    #n_samples = np.min([logits.size(0), 16])
    n_logits = logits.size(1)
    n_vocab = logits.size(-1)


    label_names = [f'N_Once[{i}]' for i in range(n_logits)]
    colour_labels = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple'] * 2

    vocab_16 = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    #plotting_logits = [logits[idx, :, :] for idx in range(n_samples)]

    scale_fig = (1 if n_logits == 4 else 2)

    fig, axes = plt.subplots(n_logits, 1, figsize=(16, 11*scale_fig), clear=True, dpi=100)
    plt.rcParams.update({'font.size': 9})
    plt.title(f'Nonce Logit Probability per Class')
    plt.xlabel('Token Index')
    plt.ylabel('Probability')
    plt.grid(True)
    axes = axes.flatten()
    batch_item_ = logits[0]
    logit_data = batch_item_.to(dtype=torch.float32).cpu().numpy()


    y_max = 1.25 * np.max([logit_data[i].max() for i in range(n_logits)])
    y_min = 0.75 * np.min([logit_data[i].min() for i in range(n_logits)])
    if predicted is None:
        predicted = torch.argmax(logits[0], dim=-1).cpu().numpy()
    else:
        predicted = predicted.clone().to(dtype=torch.long).cpu().numpy()

    for i in range(n_logits):
        ax = axes[i]
        ax.axis('on')
        if solved is not None:
            solutuion = solved.cpu().numpy()
            axes[i].axvline(x=solutuion[i], color="000000", linestyle='-', linewidth=1.2,
                            label=f'Known Token={ tknzr.detokenize([solutuion[i]])  }', antialiased=True, snap=True)
            axes[i].axhline(y=logit_data[i, solutuion[i]], color="000000", linestyle='-', linewidth=1.2,
                            label=f'Prediction Prob={logit_data[i, solutuion[i]]:.5f}', antialiased=True, snap=True)

            for j in range(n_logits):
                if j != i:
                    axes[i].axvline(x=solutuion[j], color=colour_labels[j],
                                    linestyle='-', linewidth=1., antialiased=True, snap=True)

        axes[i].axvline(x=predicted[i], color="000000", linestyle=':', linewidth=1.95,
                        label=f'Predicted Token={ tknzr.detokenize( [predicted[i]]) }', antialiased=True, snap=True)
        axes[i].axhline(y=logit_data[i, predicted[i]], color="000000", linestyle=':', linewidth=1.95,
                        label=f'Prediction Prob={logit_data[i, predicted[i]]:.5f}', antialiased=True, snap=True)
                        
        axes[i].step(np.arange(n_vocab), logit_data[i, :], colour_labels[i],
                     where='mid', label=label_names[i], linewidth=1, antialiased=True, snap=True)
        patch = StepPatch(values=logit_data[i, :], edges=(np.arange(n_vocab+1)-0.5), baseline=logit_data[i, :].mean(),
                          color=colour_labels[i], alpha=0.45, antialiased=True, snap=True)
        axes[i].add_patch(patch)

        if n_vocab==256:
            x_ticks = np.arange(0, n_vocab, n_vocab//32)
        else:
            x_ticks = np.arange(0, n_vocab, n_vocab//16)
        
        axes[i].set_xticks(x_ticks, labels=[tknzr.detokenize([tkn_id]) for tkn_id in x_ticks])
        try:
            axes[i].set_yticks(np.arange(y_min, y_max, (y_max-y_min)/8))
        except:
            print(f"{y_max = } {y_min = } ")
            print(f"{logits = }")
            print(f"{torch.isnan(logits).flatten().sum() = }")
            axes[i].set_yticks(np.arange(y_min, y_max+1.0e-9 , (y_max + 1.0e-9  - y_min)/8))
            
        
        #axes[i].grid(True)
        axes[i].grid(axis='x', color='0.75')
        axes[i].grid(axis='y', color='0.75')
        axes[i].legend(fontsize=8)
        """
        sns.histplot(x=np.arange(256),
                     y=logit_data[i,:],
                     bins=n_bins,
                     stat='probability',
                     label=label_names[i],
                     ax=axes[i],
                     element='step',
                     fill=True,
                     color=colour_labels[i])
        """
    fig.tight_layout()



    folder = f'logit_prob/'
    try:
        os.mkdir(filepath + folder)
    except:
        pass
    plt.savefig(filepath + folder + f'{filename}', dpi=150)
    del logits, predicted, solutuion
    gc.collect()
    if return_figure:
        return fig

    ax.clear()
    plt.rcParams.update({'font.size': 9})
    plt.close('all')
    gc.collect()
    return

#----------------------------------------------------------------
def PlotLogitPixels( logits,
                     solved=None,
                     predicted=None,
                     filepath=IMAGES_FILE_PATH,
                     filename=f'',
                     return_figure=False,
                     prenormalized=True):
    #NOTE: (batch_size, vocab_dim, n_logits = logits.shape)
    logits = torch.clone(logits.detach())
    b_, vocab_, digits_ = logits.shape
    colour_map = matplotlib.colormaps['gnuplot']
    colour_map = matplotlib.colormaps['terrain']
    label_names  = ['Nonce[0]', 'Nonce[1]', 'Nonce[2]', 'Nonce[3]']
    colour_labels= ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
    n_bins=256

    inner =  4 *4
    outer =  9 *4 *2

    if not prenormalized:
        logits = F.softmax(logits, dim=1)


    logits = logits.permute(0,2,1)
    logits = dezeroize(logits, full=True)

    n_samples = np.min([logits.size(0), 16])

    plotting_logits = [logits[idx,:,:] for idx in range(n_samples)]

    num_images = len(plotting_logits)
    num_rows = int(num_images ** 0.5)
    num_cols = (num_images + num_rows - 1) // num_rows


    #
    fig, axes = plt.subplots( 16, 1, figsize=(16,9), clear=True,  dpi=300) #layout="constrained", figsize=(8, 10),
    plt.rcParams.update({'font.size': 0})
    axes = axes.flatten()
    for i, batch_item in enumerate( plotting_logits ):
        ax = axes[i]
        if predicted is None:
            print(f'softmax used')
            predicted =  max_token_select( batch_item, soft_max=(~prenormalized))
        else:
            pred_xy = torch.stack([predicted[i], torch.arange(digits_)])
            pred_xy = pred_xy.tolist()

        batch_item_ = dezeroize(batch_item, full=False)
        data = batch_item_.to(dtype=torch.float32).cpu().numpy()
        cmap = plt.cm.get_cmap('viridis', 256)

        # Set the color values in the pixel array

        for y in range(4):
            d = np.ones((4,256,))* np.nan
            d[y,:] = data[y,:]
            p = ax.pcolor(d, norm=colors.LogNorm(vmin=np.min(d[y,:]), vmax=np.max(d[y,:])), cmap=cmap)
            ax.axis('off')
        ax.axis('off')

        if solved is None :
            # Dots where generated nonce values are
            pp = ax.scatter( *(0.5+np.array(predicted)), color='none', marker='X', s=inner*3, linewidth=1.5, edgecolors='black', alpha=1)

            #pp = ax.scatter( *(0.5+np.array(predicted)), color='black', s=outer )
            #ppp = ax.scatter(  *(0.5+np.array(predicted)), color='white', s=inner)

        if solved is not None :
            logit_index = (0.5+solved[i].cpu()).tolist()
            digit_index = (0.5+torch.arange(4)).tolist()
            nonce_xy = [xy for xy in zip(solved[i].cpu().tolist(), torch.arange(4).tolist())]

            correct_tokens = []
            tokens_xy = [xy for xy in zip(*pred_xy)]
            for nonce_digit in nonce_xy:
                if nonce_digit in tokens_xy:
                    correct_tokens += [nonce_digit]
            
            
            # Dots where the solved nonce values are
            pp = ax.scatter(logit_index, digit_index,  color='none', s=outer*0.65, linewidth=1.75, edgecolors='r', alpha=1)

            # Dots where generated nonce values are
            ppp = ax.scatter( *(0.5+np.array(predicted)), color='none', marker='X', s=inner*3, linewidth=1.5, edgecolors='black', alpha=1)

            # Dots where the generated tokens match solved nonce values
            if correct_tokens != []:
                correct_tokens = (0.5+ np.array(correct_tokens))
                pppp = ax.scatter( correct_tokens[:,0],correct_tokens[:,1], color='none', s=outer*1.5,  marker='*', linewidth=2, edgecolors='black', alpha=1)

    ax.axis('off')
    fig.tight_layout()

    if return_figure:
        return fig

    folder= f'logit_pixels/'
    try:
        os.mkdir(filepath + folder)
    except:
        pass
    plt.savefig( filepath + folder + f'(LogitPixels){filename}', dpi=300)
    ax.clear()
    plt.close('all')
    gc.collect()
    plt.rcParams.update({'font.size':   9})
    return


