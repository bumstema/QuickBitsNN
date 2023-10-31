################################################################
import numpy as np
import torch

from typing             import List, Dict
from dataclasses        import dataclass, asdict, field

from ..data_io.utils        import get_device
from ..framework.constants  import START_TOKEN, END_TOKEN, PADDING_TOKEN, INDEX_TO_HASHEX, HASHEX_VOCABULARY


#  Encode Bytes as tokens for Hashex Dictionary.
#--------------------------------------------------------------
#--------------------------------------------------------------
class Tokenizer:
    def __init__(self):
        self.token2index = {START_TOKEN: 256, END_TOKEN:257}
        self.index2token = {256: START_TOKEN, 257: END_TOKEN}
        self.vocab_size = 0
        for first_letter in HASHEX_VOCABULARY:
            for second_letter in HASHEX_VOCABULARY:
                hashex_token = f'{first_letter}{second_letter}'
                hashex_index = int(hashex_token,16)
                if hashex_token not in self.token2index:
                    self.token2index[hashex_token] = hashex_index
                    self.index2token[hashex_index] = hashex_token
                    self.vocab_size += 1



    def show_tokens(self):
        return self.token2index



    def fit(self, sequences):
        for sequence in sequences:
            for token in sequence:
                if token not in self.token2index:
                    self.token2index[token] = self.vocab_size
                    self.index2token[self.vocab_size] = token
                    self.vocab_size += 1



    def tokenize(self, sequence, start_token=False, end_token=False):
        seq = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
        if start_token:
            tokenized_sequence = np.concatenate(
                [np.array([self.token2index[START_TOKEN]]), np.array([self.token2index[token] for token in seq])])
        elif end_token:
                tokenized_sequence = np.concatenate(
                    [ np.array([self.token2index[token] for token in seq]), np.array([self.token2index[END_TOKEN]])])
        else:
            tokenized_sequence = np.array([self.token2index[token] for token in seq])
        return torch.tensor(tokenized_sequence, requires_grad=False)



    def detokenize(self, sequence):
        if isinstance(sequence, torch.Tensor):
            try:
                return "".join([self.index2token[token_index] for token_index in (sequence.tolist())])
            except:
                print(f"{sequence.tolist() = }")
        else:
            return "".join([self.index2token[token_index] for token_index in sequence])

    
    
    
