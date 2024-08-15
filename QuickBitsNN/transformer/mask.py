
import numpy as np
import torch
import torch.nn as nn

from ..data_io.utils import get_device
#from ..framework.tokenizer import Tokenizer


class Masks(nn.Module):
    """
    Returns a tensor of shape: [1, seq_len, embed_len]
    """
    def __init__(self,  num_heads=0, encoder_input_dim=76, decoder_input_dim=4, vocab_size=256 ):
        super().__init__()
        self.device = get_device()
        #self.device = 'cuda'
        #self.device='mps'

        self.encoder_input_dim = encoder_input_dim
        self.decoder_input_dim = decoder_input_dim
        self.vocab_size = vocab_size #Tokenizer().vocab_size
        self.num_heads = num_heads

        """
        #  All unmasked places are zero, masked items are ones
        self.look_ahead_mask = [[False, True, True, True],
                                [False, False, True, True],
                                [False, False, False, True],
                                [False, False, False, False]]

        self.null_padding_mask = [False, False, False, False]

        ### --- src[-1]
        padded_look_ahead_mask_last = [[False, True, True, False],
                                        [False, False, True, False],
                                        [False, False, False, False],
                                        [False, False, False, False]]
        padding_mask_last = [False, False, False, True]

        ### --- src[-2:]
        padded_look_ahead_mask_second_last = [[False, True, False, False],
                                        [False, False, False, False],
                                        [False, False, False, False],
                                        [False, False, False, False]]
        padding_mask_second_last = [False, False, True, True]

        ### --- src[-3:]
        padded_look_ahead_mask_third_last = [[False, False, False, False],
                                        [False, False, False, False],
                                        [False, False, False, False],
                                        [False, False, False, False]]
        padding_mask_third_last = [False, True, True, True]
        """


    # ---------------------------------------------------------
    # ---------------------------------------------------------
    def _look_ahead_mask(self, batch_size=1, solve_digit=0, single_digit_inference=False, size=4):
        #  All unmasked places are zero, masked items are ones
        mask = (torch.triu(
            torch.ones(batch_size*self.num_heads, size, size, dtype=torch.bool, device=self.device)) == 1).transpose(1, 2)#.float()
        mask = ~mask.to(dtype=torch.bool)
        mask.requires_grad_(requires_grad=False)
        if solve_digit < 1: return mask
        # Remove Look Ahead Mask for key-padded masked positions
        for i in range(solve_digit, size, 1):
            mask[:, :, i] = torch.zeros(batch_size * self.num_heads, 1, dtype=torch.bool, device=self.device)
        mask.requires_grad_(requires_grad=False)

        if single_digit_inference:
            ahead_mask = torch.zeros_like(mask)
            #ahead_mask[:,:,solve_digit-1] = mask[:,:,solve_digit-1]
            mask = ahead_mask
        return mask

    #
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def _target_padding_mask(self, batch_size=1, solve_digit=0, single_digit_inference=False, size=4):
        #  All unmasked places are zero, masked items are ones
        target_padding_mask = torch.zeros(batch_size, size, dtype=torch.bool, device=self.device)
        target_padding_mask.requires_grad_(requires_grad=False)
        if solve_digit < 1: return target_padding_mask

        offset = (size - solve_digit)
        target_padding_mask[:, solve_digit:] = torch.ones(batch_size, offset, dtype=torch.bool, device=self.device)
        target_padding_mask.requires_grad_(requires_grad=False)

        if single_digit_inference:
            pad_mask = torch.ones_like(target_padding_mask)

            pad_mask[:, solve_digit-1] = torch.zeros_like(target_padding_mask[:, solve_digit-1])
            target_padding_mask = pad_mask
        return target_padding_mask

    #
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    def generate_masks(self, batch_size=1, solve_digit=0, single_digit_inference=False, sequence_length=4) -> tuple:
        """ Returns Tuple (LookAheadMask, PaddingMask) """
        look = self._look_ahead_mask(batch_size=batch_size,
                                     solve_digit=solve_digit,
                                     single_digit_inference=single_digit_inference,
                                     size=sequence_length)

        pad = self._target_padding_mask(batch_size=batch_size,
                                        solve_digit=solve_digit,
                                        single_digit_inference=single_digit_inference,
                                        size=sequence_length)
        return (look, pad)



from typing import Optional, Any, Union, Callable
from torch import Tensor

def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

def _generate_square_subsequent_mask(
        sz: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]





if __name__ == '__main__':
    masks = Masks(num_heads=1)
    
    look_ahead_masks, padding_masks = masks.generate_masks(batch_size=1, solve_digit=0, sequence_length=76)
    
    
    is_causal_mask = _detect_is_causal_mask(look_ahead_masks )
    print(f"{is_causal_mask =}")
    
    c_mask = _generate_square_subsequent_mask(76)
    print(f"{c_mask.to(dtype=torch.bool) = }")
    print(f"{look_ahead_masks[0] = }")
    
    exit()
    
    #look_ahead_masks = [ masks._look_ahead_mask(solve_digit=idx_) for idx_ in range(5)]
    #padding_masks = [ masks._target_padding_mask(solve_digit=idx_) for idx_ in range(5)]
    for idx_ in range(76):
        print(f"{idx_} : {look_ahead_masks[0][idx_]}")
    print(f"{idx_} : {padding_masks[0] = }")
        
    '''
    look_ahead_masks = [ masks._look_ahead_mask(solve_digit=idx_, single_digit_inference=True) for idx_ in range(5)]
    padding_masks = [ masks._target_padding_mask(solve_digit=idx_, single_digit_inference=True) for idx_ in range(5)]
    for idx_ in range(5):
        print(f"{idx_} : {look_ahead_masks[idx_][0]}")
        print(f"{idx_} : {padding_masks[idx_] = }")



    all_masks = [masks.generate_masks(batch_size=5,
                                          solve_digit=(n_idx_ + 1),
                                          single_digit_inference=True) for n_idx_ in range(4)]
    look_ahead_mask, target_padding_mask = list(zip(*all_masks))
    #print(look_ahead_mask)
    #print(len(look_ahead_mask))
    #print(target_padding_mask)
    #print(len(target_padding_mask))
    '''