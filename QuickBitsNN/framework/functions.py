################################################################
import json
import hashlib
import binascii
import numpy as np
import torch
import itertools
from typing import List, Dict, Any
import base64
from scipy.optimize import curve_fit, newton
from QuickBitsNN.framework.tokenizer import Tokenizer
from QuickBitsNN.framework.labeler import Labeler


# ----------------------------------------------------------------
# ----------------------------------------------------------------

def encode_string( word ):
    return base64.b64encode(f'{word}'.encode("UTF-8","strict"))

def decode_string( word ):
    return (base64.b64decode( word )).decode("UTF-8","strict")

def little_endian(value):
    return binascii.hexlify(binascii.unhexlify(value)[::-1])

def big_endian(hash):
    return str(binascii.hexlify(binascii.unhexlify(hash)[::-1]), "ascii")

def four_byte_hex(number):
    return hex(int(0x100000000) + number)[-8:]

def int_to_tokenable_hex(number):
    return little_endian(four_byte_hex(number)).decode("utf-8")

def little_endian_hex_to_int(hex_string):
    """Decodes the integer value from the preblock hex representation."""
    # Reverse the little-endian hex string by bytes
    reversed_hex = "".join(
        reversed([hex_string[i : i + 2] for i in range(0, len(hex_string), 2)])
    )

    # Convert the reversed hex string to base 10 integer
    base_10_integer = int(reversed_hex, 16)
    return base_10_integer


# ----------------------------------------------------------------
def HeaderNonceMiningHash(prenonce_header, nonce):
    header = prenonce_header + nonce
    if len(header) != 160:
        print(f"{prenonce_header = }  {nonce = } {len(header) = }")
        exit()
    header = binascii.unhexlify(header)
    hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
    hash = binascii.hexlify(hash)
    hash = binascii.hexlify(binascii.unhexlify(hash)[::-1])
    return hash.decode("utf-8")


# ----------------------------------------------------------------
def construct_block_header(
    version, prev_block_hash, merkle_root, timestamp, difficulty_target, nonce
):
    block_header = (
        version.to_bytes(4, byteorder="little")
        + bytes.fromhex(prev_block_hash)
        + bytes.fromhex(merkle_root)
        + int(timestamp).to_bytes(4, byteorder="little")
        + int(difficulty_target).to_bytes(4, byteorder="little")
        + int(nonce).to_bytes(4, byteorder="little")
    )
    return block_header


# ----------------------------------------------------------------
def LeadingZeros(input_string: str) -> int:
    zeros = lambda s: len(s) - len(s.lstrip("0"))
    return zeros(input_string)


# ----------------------------------------------------------------
def BitsZeros(bits):
    """Decodes the required leading zeros from the bits parameter."""
    with torch.autocast(device_type="cuda", enabled=False):
        hex_bits = four_byte_hex(int(bits))
        bits_exponent = str(hex_bits)[:2]
        bits_exponent = int(f"0x" + bits_exponent, 16)
        required_zeros = 2 * (32 - bits_exponent)
        return required_zeros

# ----------------------------------------------------------------
def LogitRankingsFromKnownSolutions__(batched_logits, batched_nonce):
    """Selects how far from the top ranking each nonce digit token is within the generated logits."""
    logitem_ranks = torch.argsort(batched_logits, dim=1)
    solved_logit_rankings = [
        [
            logitem_ranks[b_idx_, logitem_idx_, n_idx_].item()
            for n_idx_, logitem_idx_ in enumerate(nonce_)
        ]
        for b_idx_, nonce_ in enumerate(batched_nonce.unbind())
    ]
    return solved_logit_rankings


# ----------------------------------------------------------------
def LogitRankingsFromKnownSolutions(batched_logits, batched_nonce):
    """Selects how far from the top ranking each nonce digit token is within the generated logits."""
    """ batched_logits=[batch, vocab, digits], batched_nonce=[batch, digits]  """
    """ Returns: logit_rankings.shape = [batch_, vocab_, digits_] """
    batch_digit_logits =  batched_logits.permute(0,2,1)
    logit_rankings = [ get_rank(digit_logit_, batched_nonce[idx_]) for idx_, digit_logit_ in enumerate(batch_digit_logits.unbind()) ]
    return logit_rankings

    
# ----------------------------------------------------------------
def get_rank(x, indices):
    """Input:(x=[batch,logit], indices=[batch,targets])"""
    vals = x[range(len(x)), indices]
    return (x > vals[:, None]).long().sum(1).cpu().tolist()


# ----------------------------------------------------------------
"""
def probability_function_for_logit_ranks(batched_logits, batched_nonce_tokens, reply_all_ranks=False):
    #  return rank_prob_per_logit.shape = [n_logits, vocab_size] 
    b_, n_logits_ = batched_nonce_tokens.shape
    solved_digit_rankings = LogitRankingsFromKnownSolutions(batched_logits, batched_nonce_tokens)
    rank_prob_per_logit = [np.histogram(np.array(logit_ranks_), bins=256, density=True)[0] for logit_ranks_ in solved_digit_rankings]
    rank_prob_per_logit = np.array(rank_prob_per_logit)
    rank_prob_per_logit = torch.tensor(rank_prob_per_logit, dtype=torch.float32)
    #rank_prob_per_logit = torch.stack([torch.tensor(rank_prob_per_logit[n_]) for n_ in range(n_logits_)],dim=0)

    if reply_all_ranks:
        return (rank_prob_per_logit.to(device=batched_logits.device), solved_digit_rankings)

    return rank_prob_per_logit.to(device=batched_logits.device)
"""

# ----------------------------------------------------------------
def assign_probability_to_logit_class_from_rank(batched_logits, logit_ranking_probs):
    b_, n_logits, vocab_  = batched_logits.shape
    logitem_ranks = torch.argsort(batched_logits, dim=1)
    class_rank_probabilities = torch.zeros_like(batched_logits)

    for batch_item_ in range(b_):
        for logit_idx_ in range(n_logits):
            ranks_by_class_index = logitem_ranks[batch_item_, logit_idx_, :]
            reset_probs = logit_ranking_probs[logit_idx_][ranks_by_class_index]
            class_rank_probabilities[batch_item_, logit_idx_, :] = reset_probs

    return class_rank_probabilities


# ----------------------------------------------------------------
def Nonce_Permutations( nonce_letter_lists ):
    """ nonce_letter_lists = [ [digit_0], [digit_1], [digit_2], [digit_3] ]  """
    """ possible_nonces = [ [nonce_0], [nonce_1], .... ] """
    possible_nonces = [ l for l in list(itertools.product( *nonce_letter_lists ))]
    return possible_nonces
    #


# ----------------------------------------------------------------
def del_dups(mylist_):
    return list(dict.fromkeys(mylist_))
#

# -----------------------------------------------------------------------------------
def generate_nonce( logits, samples=1, top_k=1, top_p=0.0, from_top=True, from_rank=False, ranks=None, largest=True):
    """ batch, digit, vocab = logit.shape """
    batch_, d_ = logits.size(0), logits.size(1)
    next_token = torch.zeros((batch_, d_, samples,), device=logits.device)
    token_probabilities = None

    if from_rank :
        if ranks is None: print(f"Error! Must include rankings for logits.")
        if ranks is not None: logits = logits.to(device=ranks.device)
        token_probabilities = assign_probability_to_logit_class_from_rank(logits, ranks)


    if from_top :
        filtered_logits = torch.zeros_like(logits)
        next_token_logits = logits.clone().detach()
        for idx_ in range(d_):
            filtered_logits[:, idx_, :] = top_k_top_p_filtering(next_token_logits[:, idx_, :],
                                                                      top_k=top_k,
                                                                      top_p=top_p,
                                                                      largest=largest)  # Apply top-k and/or top-p
        token_probabilities = torch.softmax(filtered_logits, dim=-1, dtype=torch.float32)

    if token_probabilities is None:
        token_probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)

    for idx_ in range(d_):
        next_token[:, idx_, :] = torch.multinomial(token_probabilities[:, idx_], num_samples=samples)  # Sample

    return next_token.squeeze(1)


# -----------------------------------------------------------------------------------------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, largest=True ):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            :param largest: bool - select most likely (True) or least likely (False)
            :param logits:  torch.tensor with shape=[batch, vocab]
            :param top_k:   top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            :param top_p:   top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    # Define Value which causes the Result from Softmax to equal 0.0 at that Logit Class.
    filter_value = torch.tensor(-1e9).item()
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, largest=largest)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1, dtype=torch.float32), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]

        if logits.size(-1)-indices_to_remove.size(0) <3 : return logits
        logits[indices_to_remove] = filter_value
    return logits


# -----------------------------------------------------------------------------------
def multiple_nonce_candidates(list_of_moe_logits, top_k=0, top_p=0.0, samples=1, largest=True):
    """  List of N:  moe_logits.shape = batch, digit, vocab """
    """ Returns a dict: { batch_number : torch.tensor([[Nonce0],[Nonce1],...]).shape = [4, n_candidates] } """
    n_logits = len(list_of_moe_logits)
    batch_ = list_of_moe_logits[0].size(0)
    decoder_input_dim = list_of_moe_logits[0].size(1)
    current_device = list_of_moe_logits[0].device
    nonce_by_digit = []
    for idx_ in range(decoder_input_dim):
        logit_lists = [moe_logit_[:, idx_, :].clone() for moe_logit_ in list_of_moe_logits]

        three_top_ks_per_digit = [generate_nonce(logit_lists[n_].clone().unsqueeze(1),
                                                samples=samples,
                                                top_k=top_k[n_],
                                                top_p=top_p[n_],
                                                largest=largest) for n_ in range(n_logits)]
        three_top_ks_per_idx = torch.cat(three_top_ks_per_digit, dim=-1)
        nonce_by_digit += [three_top_ks_per_idx]

    top_k_tokens_per_digit = torch.stack(nonce_by_digit, dim=1).to(dtype=torch.long)
    candidate_tokens = {}
    for bdx_ in range(batch_):
        top_k_per_digit_as_list = [top_k_tokens_per_digit[bdx_, idx_].tolist() for idx_ in range(decoder_input_dim)]
        top_k_per_digit_as_list = [list(np.unique(digits_)) for digits_ in top_k_per_digit_as_list]
        nonce_tokens = Nonce_Permutations(top_k_per_digit_as_list)
        # Strip Repeated Predicted Nonce Sequences
        nonce_tokens = [tuple(x) for x in dict.fromkeys(tuple(x) for x in nonce_tokens)]
        nonce_tokens = torch.tensor(nonce_tokens, device=current_device, dtype=torch.long)
        candidate_tokens.setdefault(bdx_, nonce_tokens)

    return candidate_tokens


# ===================================================
def restructure_generated_nonces_for_eq_tokens(candidate_tokens):
    """ Reshaping dict into tensor of shape: [batch, trials, digits] """
    tokens = [candidate_tokens[idx_] for idx_ in sorted(candidate_tokens.keys())]
    tokens = torch.stack(tokens, dim=0)
    return tokens


# =====================================
def nonce_token_to_int( tokens):
    """ Input Tokens:  (torch.tensor size (sequence) )"""
    tokenizr = Tokenizer()
    labelr = Labeler()
    l_e_hex = tokenizr.detokenize(tokens)
    nonce_as_int = labelr._little_endian_hex_to_int(l_e_hex)
    return nonce_as_int


# =================================================================================
def check_hashes_from_generated_nonce(single_block_header, predicted_tokens):
    """ Input Tokens: """
    labelr = Labeler()
    n_nonces = 1
    accepted_nonces = None
    #full_pred_tokens = torch.cat([single_block_header.detach().repeat(n_nonces, 1), predicted_tokens.unsqueeze(0).detach()], dim=1)
    full_header_tokens = torch.cat([single_block_header.detach(), predicted_tokens.detach()], dim=0)
    ok_hashes = labelr(full_header_tokens.unsqueeze(0), from_tokens=True, reply_with_acceptance=True)
    if (True in ok_hashes):
        return [predicted_tokens]
        #accepted_nonces = [ t for t in predicted_tokens[ok_hashes].unbind()]
    del labelr
    del full_header_tokens, ok_hashes
    return accepted_nonces



# =================================================================================
def logit_classes_above_random_chance( logits, labels):
    """ Computes the probability for each logit, then for every digit, check the prob for class at label,
        then return the total number where softmax prob is above random chance (1/vocab).

    input: logits.shape = [batch, vocab, digits]
                labels.shape = [batch, digits]

    output: above_rand.shape = torch.tensor([digits])
    """
    #mean_probability = (all_logits_probability.view(-1).mean(dim=0)).item()
    n_tokens_above_random_by_class = torch.zeros_like( labels[0], dtype=torch.long )
    prob_for_random = 1./logits.size(1)
    prob_of_logit_at_label = []
    all_logits_probability = torch.softmax(logits, dim=1, dtype=torch.float32)
    for idx_, class_labels in enumerate(labels.unbind()):
        item_prob = [all_logits_probability[idx_, known_label_, digit_] for digit_, known_label_ in enumerate(class_labels)]
        mean_prob = torch.mean(all_logits_probability[idx_, :, :], dim=0)
        n_tokens_above_random_by_class += (torch.tensor(item_prob, device=logits.device) >= mean_prob)
    return n_tokens_above_random_by_class.to(dtype=torch.long)
 

# =================================================================================
def negative_tan(x, a, b, yO):
    return -np.tan((a*x) + b) + yO


# =================================================================================
def curve_fit_ranks_to_negtan( probability_at_rank:torch.Tensor ) -> Dict :
    """ Assuming probability_at_rank is a torch.Tensor
    Returns: {top_k, top_p}
    """
    n_classes = len(probability_at_rank)
    x_data = np.arange(1,n_classes+1)
    y_data = np.array(probability_at_rank)

    # Perform curve fitting
    popt, pcov = curve_fit(negative_tan, x_data, y_data)
    # Extract optimal parameters
    a_opt, b_opt, yO_opt = popt

    #x_data = np.linspace(1, n_classes, 512)
    y_data = np.array( [ negative_tan(x, a_opt, b_opt, yO_opt) for x in x_data ] )
    return (x_data, y_data)


# =================================================================================
def estimate_rank_at_half_probability( probability_at_rank:torch.Tensor ) -> Dict :
    """ Assuming probability_at_rank is a torch.Tensor
    Returns: {top_k, top_p}
    """
    n_classes = len(probability_at_rank)
    x_data = np.arange(1,n_classes+1)
    y_data = probability_at_rank.numpy()

    # Perform curve fitting
    popt, pcov = curve_fit(negative_tan, x_data, y_data)

    # Extract optimal parameters
    a_opt, b_opt, yO_opt = popt

    # Given y value
    target_y = 1./256

    root_function = lambda x: negative_tan(x, a_opt, b_opt, yO_opt) - target_y

    # Use bisection method to find the root
    #root = newton(root_function, 256/2., args=(target_y,))
    root = newton(root_function, 256/2.)
    mid_point = int(root // 1)
    
    cumulative_prob = torch.sum( probability_at_rank[:mid_point] )
    
    
    return {'top_k':mid_point, 'top_p':cumulative_prob}

# =================================================================================
