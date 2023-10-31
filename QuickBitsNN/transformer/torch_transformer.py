import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict

from ..data_io.utils import get_device



#----------------------------------------------------------------
@dataclass
class ModelConfig:
    num_tokens  :int
    encoder_dim   :int
    num_heads   :int
    num_encoder_layers :int
    num_decoder_layers :int
    dim_feedforward :int
    dropout_p   :float

    num_tokens  = 256
    encoder_dim = 256
    num_heads   = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 256
    dropout_p   = (0.2)
    


    



#----------------------------------------------------------------
#                       '📏'
#----------------------------------------------------------------
class Torch_Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        encoder_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout_p,
        **kwargs
    ):
        super().__init__()
        self.device = get_device()
        self.model_type = "Torch_Transformer"
        self.encoder_dim = encoder_dim
        self.embedding = nn.Embedding(num_tokens, encoder_dim, device=get_device())
        #self.target_embedding = nn.Embedding(num_tokens, encoder_dim, device=get_device())
        self.positional_encoding = Positional_Encodings(num_tokens, self.encoder_dim)
        
        self.transformer = nn.Transformer(
            d_model=encoder_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_p,
            activation='gelu',
            device=self.device,
            batch_first=True
        )

        self.hashex_logits = nn.Conv2d(76, 4, 1, stride=1, padding=0, device=get_device())

        for params in self.transformer.parameters():
            if params.dim() > 1:
                #nn.init.normal_(params)
                nn.init.xavier_normal_(params)


    @torch.autocast(device_type="cuda")
    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):

        # Define src to be of only 1 channel [N, C, w]
        #  thus it passes C channels into forward N times

        tgt_in_size = tgt.size(-1)
        src_in_size = src.size(-1)
        
        mask = (torch.triu(torch.ones(tgt_in_size, tgt_in_size)) == 1).transpose(0, 1).float()
        look_ahead_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        look_ahead_mask.to(self.device)


        # Embedding + positional encoding - Out size = (batch_size, sequence length, encoder_dim)
    
        src = self.embedding(src.squeeze(1)) + self.positional_encoding(src_in_size)
        tgt = self.embedding(tgt.squeeze(1)) + self.positional_encoding(tgt_in_size)


        out = self.transformer(src.squeeze(0), tgt.squeeze(0), tgt_mask=look_ahead_mask)
        
        out = self.hashex_logits(out.unsqueeze(-1))
        out = out.squeeze(-1)
 

        return out
      
      
#----------------------------------------------------------------
class Positional_Encodings(nn.Module):
    def __init__(self,  seq_len, embed_len):
        super(Positional_Encodings, self).__init__()
        self.device = get_device()
        self.pe = torch.zeros(seq_len, embed_len, device=self.device)
        position = torch.arange(0, seq_len, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_len, 2, device=self.device) * (-np.log(10000.0)/ embed_len))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)


    @torch.autocast(device_type="cuda")
    def forward(self, x_len):
        return self.pe[:x_len, :]


