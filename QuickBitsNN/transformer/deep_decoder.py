import typing
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from dataclasses import dataclass, asdict

from .mask import Masks
from .positional_encoding import Positional_Encoding
from ..data_io.utils import get_device
from ..data_io.show import pixelize
from ..framework.tokenizer import Tokenizer




#----------------------------------------------------------------
@dataclass
class DecoderModelConfig:
    vocab_size: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    dropout_p: float

    vocab_size = Tokenizer().vocab_size
    embedding_dim = 512                 # LLAMA3-emb    : 4096
    num_heads = 8                       # LLAMA3-heads  : 32
    num_layers = 2                      # LLAMA3-layers : 32
    dim_feedforward = 4*embedding_dim   # LLAMA3-ff     : 4x emb
    dropout_p = 0.00


#----------------------------------------------------------------
#           Transformer Model
#----------------------------------------------------------------
class DeepDecoder(nn.Module):
    """
    Transformer Decoder utilizes the output from a Transformer Encoder to generate logits.
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dim_feedforward, dropout_p, device=None):
        super().__init__()
        self.device=get_device()
        self.model_type = 'TransformerDecoderStack'
        self.encoder_input_dim = 80
        self.decoder_input_dim = 4
        self.block_tail_dim = 4
        self.vocab_size = vocab_size
        
        self.num_layers = num_layers
        self.encoder_num_layers = 1
        self.decoder_num_layers = num_layers - self.encoder_num_layers

        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=dropout_p)
        #
        
        self.temp = 1.0
        self.top_p_prob = 0.0666
        #

        self.embedding_dim = embedding_dim
        self.combined_emb_dim = self.embedding_dim
        self.sqrt_dim_embed = np.sqrt(self.embedding_dim)
        self.starting_token_id = 256
        self.padding_token_id = 257

        self.embed_norm = RMSNorm(self.embedding_dim)
        pe_size = np.max([self.vocab_size, self.embedding_dim, self.encoder_input_dim, self.combined_emb_dim ])
        self.positional_enc = Positional_Encoding(pe_size, pe_size)
      
        #
        #
        self.mask = Masks(num_heads=self.num_heads,
                            encoder_input_dim=self.encoder_input_dim,
                            decoder_input_dim=self.decoder_input_dim,
                            vocab_size=self.vocab_size)
        #

        self.autoregression = True
        self.random_target = False
        self.single_digit_inference = False
        self.quicknorm = lambda x:  (x - torch.mean(x, dim=[1,2], keepdim=True)) / torch.std(x, dim=[1,2], keepdim=True)

        #
        # Block Encoder
        # ------------------------------------------------------------
        self.block_encoder_embedding = nn.Embedding( (self.vocab_size+2), self.embedding_dim, padding_idx=self.padding_token_id, device=self.device)
        
        self.encoder_transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_p,
            activation = torch.nn.SiLU(),
            batch_first = True,
            bias=False,
            device = self.device
        )
        

        
        self.block_encoder_model = nn.TransformerEncoder(
            self.encoder_transformer_layer,
            num_layers = self.encoder_num_layers,
            #norm = LayeRMSNorm( self.encoder_input_dim, self.embedding_dim ),
            norm = nn.LayerNorm([self.encoder_input_dim, self.embedding_dim], elementwise_affine=False, bias=False),
            enable_nested_tensor=False
            )
        
        
        #
        # Decoder Stack
        # ------------------------------------------------------------
        self.decoder_embedding = nn.Embedding( (self.vocab_size+2), self.combined_emb_dim, padding_idx=self.padding_token_id, device=self.device)

        self.decoder_transformer_layer = nn.TransformerDecoderLayer(
            d_model = self.combined_emb_dim,
            nhead = self.num_heads,
            dim_feedforward = self.dim_feedforward,
            dropout = self.dropout_p,
            activation = torch.nn.SiLU(),
            batch_first = True,
            device = self.device)
            

        
        self.block_decoder_model = nn.TransformerDecoder(
            self.decoder_transformer_layer,
            num_layers = self.decoder_num_layers,
            norm = nn.LayerNorm([self.decoder_input_dim, self.embedding_dim], elementwise_affine=False, bias=False )
            )

        self.normalize_decoder_output =  nn.LayerNorm( [self.decoder_input_dim, self.embedding_dim], device=self.device, dtype=torch.float32, elementwise_affine=True, bias=False )
        self.decoder_logit_head = nn.Linear(self.combined_emb_dim, self.vocab_size, bias=True, device=self.device)

        #
        #
        
        #self.vq = VectorQuantizer(embedding_dim=self.encoder_input_dim, num_embeddings=self.vocab_size)
        #self.vq.requires_grad_(requires_grad=False)
        
    # ===============================
    def _update_dropout(self, new_p):
        n_enc_layers = len( self.block_encoder_model.layers )
        for i in range(n_enc_layers):
            self.block_encoder_model.layers[i].dropout = nn.Dropout(p=new_p)
            self.block_encoder_model.layers[i].dropout1 = nn.Dropout(p=new_p)
            self.block_encoder_model.layers[i].dropout2 = nn.Dropout(p=new_p)

        n_dec_layers = len( self.block_decoder_model.layers )
        for i in range(n_dec_layers):
            self.block_decoder_model.layers[i].dropout = nn.Dropout(p=new_p)
            self.block_decoder_model.layers[i].dropout1 = nn.Dropout(p=new_p)
            self.block_decoder_model.layers[i].dropout2 = nn.Dropout(p=new_p)
            self.block_decoder_model.layers[i].dropout3 = nn.Dropout(p=new_p)
        

    # ================================================================================================
    def _apply_start_of_sequence(self, tgt: torch.Tensor, train=False, masked_idx=3) -> torch.Tensor:
        
        if train :
            cls_token = torch.clone(tgt)
            cls_token[:, 1:] = tgt[:, :-1]
            cls_token[:, 0] = self.starting_token_id * torch.ones_like(tgt[:, 0])
            return cls_token

        # Place Classification Token in All Positions of Sequence for Inference #
        if (not train) & self.autoregression:
            cls_token = torch.ones_like(tgt)
            cls_token[:,0] = self.starting_token_id * torch.ones_like(tgt[:,0])
            cls_token[:,1:] = self.padding_token_id * torch.ones_like(tgt[:,1:])
            cls_token.requires_grad_(requires_grad=False)
            return cls_token

        if self.random_target:
            cls_token = torch.randint(0, self.vocab_size, (tgt.shape), device=self.device, requires_grad=False)
            cls_token[:,0] = self.starting_token_id * torch.ones_like(tgt[:,0])
            return cls_token

    # ----------------------------------------------------------------------
    def _embed(self, sequence:torch.Tensor, embedding=None) -> torch.Tensor:
        b_, seq_dim_ = sequence.shape
        
        emb_seq = embedding(sequence)
        #emb_seq = emb_seq * self.sqrt_dim_embed
        emb_seq = self.quicknorm(emb_seq) * self.sqrt_dim_embed
        emb_dim_ = emb_seq.size(-1)
        
        pe = self.positional_enc(seq_dim_).repeat(b_, 1, 1)
        pe = pe[:,:seq_dim_,:emb_dim_]
        emb_seq = emb_seq + pe
        
        emb_seq.requires_grad_(requires_grad=True)
        #emb_seq = self.embed_norm(emb_seq)
        emb_seq = self.dropout(emb_seq)
        return emb_seq

    # ----------------------
    def _format_for_return(self, encoder_source, encoder_output, decoder_output, logits, selected_tokens=None, encoder_embedding=None, vq_out=None) -> dict[str, torch.Tensor]:
        logit_solution = {}
        logit_solution['encoder_output'] = encoder_output[0].permute(1,0).clone().detach()
        logit_solution['decoder_output'] = decoder_output[0].permute(1,0).clone().detach()
        logit_solution['logits'] = logits.permute(0,2,1).to(dtype=torch.float32)
        logit_solution['nonce'] = (torch.argmax(logits, dim=-1).to(dtype=torch.long) if selected_tokens is None else selected_tokens)
        logit_solution['header'] = torch.concat([encoder_source, logit_solution['nonce']], dim=1)
        if encoder_embedding is not None:
            logit_solution['pixels_x[0]'] = encoder_embedding[0].permute(1,0).clone().detach()
            
        if vq_out is not None:
            logit_solution['dictionary_loss'] = vq_out['dictionary_loss'].to(dtype=torch.float32)
            logit_solution['commitment_loss'] = vq_out['commitment_loss'].to(dtype=torch.float32)
            
        return logit_solution
        
    
    # ===================================================================================
    #               FORWARD
    # ===================================================================================
    def forward(self, encoder_source, decoder_target, train=True, masked_digit=3) -> dict[str, torch.Tensor]:


        assert encoder_source.size(0) == decoder_target.size(0)
        batch_ = decoder_target.size(0)

        if not train:
            with torch.no_grad():
                return self.inference(encoder_source, decoder_target, temperature=self.temp, top_p_prob=self.top_p_prob )


        if train:
            #----- Encoder
            encoder_mask, encoder_pad_mask = self.mask.generate_masks(batch_size=batch_,
                                                                    sequence_length=self.encoder_input_dim,
                                                                    #solve_digit=self.encoder_input_dim)
                                                                    solve_digit = (self.encoder_input_dim))
            encoder_mask = torch.zeros_like(encoder_mask, dtype=torch.bool)
            
            (encoder_output, vq_loss) = self.encoder_stack(encoder_source, encoder_mask, encoder_pad_mask, reply_vq_loss=True)


            #------ Decoder
            look_ahead_mask, target_padding_mask = self.mask.generate_masks(batch_size=batch_,
                                                                    sequence_length=self.decoder_input_dim,
                                                                    solve_digit=self.block_tail_dim)

            tgt_emb = self._apply_start_of_sequence(decoder_target, train=train)
            tgt_emb = self._embed(tgt_emb, embedding=self.decoder_embedding)
            
            logits, decoder_output = self.transformer_to_logit(tgt_emb, encoder_output, look_ahead_mask, target_padding_mask, mem_mask=encoder_mask)
            
            logit_solution = self._format_for_return(encoder_source, encoder_output, decoder_output, logits, vq_out=vq_loss)
            return logit_solution


    # -------------------------------------------------------------------------------------
    def encoder_stack(self, encoder_input, encoder_mask, encoder_pad_mask, reply_emb=False, reply_vq_loss=False):
                                                
        emb_src_b = self._embed(encoder_input, embedding=self.block_encoder_embedding )
        tail_src = self.block_encoder_model(emb_src_b, mask=encoder_mask, src_key_padding_mask=encoder_pad_mask, is_causal=True)
        encoder_output = tail_src

        #quantized_encoder_output = self.vq(encoder_output)
        #encoder_output = quantized_encoder_output['quantized_features']
        

        if reply_emb:
            embedding_output =  emb_src_b
            return (encoder_output, embedding_output)
            
        if reply_vq_loss:
            #return (encoder_output, quantized_encoder_output)
            return (encoder_output, None)
            
        return encoder_output
        
    # -------------------------------------------------------------------------------------
    def transformer_to_logit(self, emb_tgt, emb_src, look_ahead_mask, target_padding_mask, mem_mask=None):

        decoder_output = self.block_decoder_model(emb_tgt, emb_src,
                            tgt_mask=look_ahead_mask,
                            memory_mask=mem_mask,
                            tgt_key_padding_mask=target_padding_mask,
                            tgt_is_causal=True,
                            memory_is_causal=True)
        
        decoder_output = self.normalize_decoder_output(decoder_output)

        decoder_logits = self.decoder_logit_head( decoder_output )

        return decoder_logits, decoder_output

    # ===================================================================================
    #               INFERENCE
    # ===================================================================================
    def inference(self, encoder_source, tgt, temperature=0.6, top_p_prob=0.9) -> dict[str, torch.Tensor]:

        batch_ = encoder_source.size(0)
        encoder_mask, encoder_pad_mask = self.mask.generate_masks(batch_size = batch_,
                                                    sequence_length = self.encoder_input_dim,
                                                    solve_digit = (self.encoder_input_dim-self.block_tail_dim) )
        encoder_mask = torch.zeros_like(encoder_mask, dtype=torch.bool)

        (encoder_output, src_emb) = self.encoder_stack(encoder_source, encoder_mask, encoder_pad_mask, reply_emb=True)


        masks = [self.mask.generate_masks(batch_size = batch_,
                                          solve_digit = (n_idx_ + 1),
                                          single_digit_inference = self.single_digit_inference,
                                          sequence_length = self.decoder_input_dim) for n_idx_ in range(self.decoder_input_dim)]
        
        look_ahead_mask, target_padding_mask = list(zip(*masks))


        #~~~~~~~~~~ AUTO REGRESSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #
        logits = torch.zeros(batch_, self.decoder_input_dim, self.vocab_size, device=self.device)
        selected_tokens = torch.zeros(batch_, self.decoder_input_dim, device=self.device)
        #token_probabilities = torch.zeros(batch_, self.decoder_input_dim, device=self.device)
        #token_losses = torch.zeros_like(tgt)


        autorecursive_tgt = torch.clone(tgt.detach())
        autorecursive_tgt = self._apply_start_of_sequence(autorecursive_tgt, train=False)

        for n_idx_ in range(self.decoder_input_dim):

            tgt_emb = self._embed(autorecursive_tgt, embedding=self.decoder_embedding)
            batched_logits, decoder_output = self.transformer_to_logit(tgt_emb, encoder_output, look_ahead_mask[n_idx_], target_padding_mask[n_idx_], mem_mask=encoder_mask)
            
            logits[:, n_idx_, :] = torch.clone(batched_logits[:, n_idx_, :])

            if temperature > 0.0:
                token_prob = torch.softmax( (logits[:, n_idx_, :] / temperature), dim=1)
                generated_token = sample_top_p(token_prob, top_p_prob).flatten()
            else:
                # Naive Token Selection
                #generated_token = torch.multinomial(token_prob, num_samples=1)
                generated_token = torch.argmax(logits[:, n_idx_], dim=-1)
            
            
            selected_tokens[:, n_idx_] = generated_token.flatten()
            #token_probabilities[:, n_idx_] = torch.take_along_dim(token_prob, generated_token, dim=1).flatten()

            """
            if known_labels is not None:
                token_losses[:, n_idx_ ] = torch.nn.functional.cross_entropy(
                    input=logits.transpose(1, 2), #[:,:, :(n_idx_+1) ],
                    target=known_labels,#[:, :(n_idx_+1)],
                    reduction="none",
                    ignore_index = self.padding_token_id
                    )[:, n_idx_]
            """

            if self.autoregression & (n_idx_ < (self.decoder_input_dim-1) ):
                autorecursive_tgt[:, n_idx_+1] = generated_token
                
            elif (not self.autoregression):
                autorecursive_tgt[:, n_idx_] = generated_token

   
        logit_solution = self._format_for_return(encoder_source, encoder_output, decoder_output, logits, selected_tokens=selected_tokens, encoder_embedding=src_emb)
            
        return logit_solution
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        



# =================================================================
class LayeRMSNorm(torch.nn.Module):
    def __init__(self, layers: int, dim: int, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.n_layers = layers
        self.weight = nn.ParameterList([])
        self.bias = nn.ParameterList([])
        for n_ in range(self.n_layers):
            self.weight.append( nn.Parameter(torch.ones(dim)) )
        for n_ in range(self.n_layers):
            self.bias.append( nn.Parameter(torch.zeros(dim)) )

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x_out = torch.zeros_like(x)
        for n_ in range(self.n_layers):
            x_out[:,n_:n_+1,:] = (self.weight[n_] * self._norm( x[:,n_:n_+1,:].float()).type_as(x)) + self.bias[n_]
            
        return x_out

# =================================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
        
# =================================================================


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# =================================================================

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim=5, num_embeddings=256, epsilon=1.0e-7):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.
        self.device=get_device()
        # embedding_dim = seq_len
        self.embedding_dim = embedding_dim
        # num_embbedings = vocab_size
        self.num_embeddings = num_embeddings
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 256 ** 0.5
        e_i_ts = torch.ones( (embedding_dim, num_embeddings,), device=self.device)#.uniform_(-limit, limit)
        self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))
        nn.init.xavier_uniform_(e_i_ts, gain=nn.init.calculate_gain('relu'))


    # ----------------------------------------------------------------------
    def forward(self, x_in):
        #flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        
        #x = x_in['features'].clone()
        x = x_in.clone()
        # Channels Last
        flat_x = x.permute(0, 2, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        #print(f"{distances.shape = }")
        #print(f"{encoding_indices.shape = }")

        quantized_x = torch.nn.functional.embedding(
            encoding_indices.view(x.shape[0], x.shape[2]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 2, 1)

        #print(f"{quantized_x.shape = }")
        #print(f"{x.shape = }")
        # Channels First
        # See second term of Equation (3).
        dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()
    
        output_ = {'quantized_features': quantized_x, 'dictionary_loss':dictionary_loss, 'commitment_loss':commitment_loss }
        return output_
        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )
