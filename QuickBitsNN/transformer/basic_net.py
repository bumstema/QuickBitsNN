import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, asdict
from copy import deepcopy

import time
from .positional_encoding import Positional_Encoding


from ..data_io.utils import get_device
from ..data_io.show import pixelize
from ..framework.tokenizer import Tokenizer
from ..framework.labeler import Labeler


#----------------------------------------------------------------
#----------------------------------------------------------------
class Basic_Net(nn.Module):
    """
    Extra Basic NN Model to predict Nonce Token Logits
    """
    # Constructor
    def __init__(self):
        super().__init__()
        self.dtype=torch.float32
        self.device = get_device()
        self.model_type = "Basic_Net"

        self.input_seq_len = 80
        self.output_seq_len = 4
        self.vocab_size = 256
        self.embedding_dim = 256

        #self.dropout = nn.Dropout(p=0.0025)
        self.dropout = nn.Dropout(p=0.00)
        self.activate = nn.LeakyReLU()
        self.embedding = EmbeddingMethod(depth_=True, embed_dim_=self.embedding_dim, seq_len_=self.input_seq_len )
        self.featureifier = ConvNet()
        self.vq = VectorQuantizer()
        self.reconstruction = ReconstructionNet()

        #self.classifier = LinearClassifier()
        
        self.single_digit_inference=False
    
    # -------------------------------------------------------------------------------
    def freeze(self, features=False, classifier=False):
        self.featureifier.requires_grad_(requires_grad=features)
        self.classifier.requires_grad_(requires_grad=classifier)
        


    # ===================================================================================
    #                           FORWARD
    # ===================================================================================
    def forward(self, encoder_source, decoder_target, train=False, masked_digit=3) -> dict[str, torch.Tensor]:

        logit_solution = {}
        
        x = self.embedding(encoder_source, tgt=decoder_target, train=train)
        
        logit_solution['pixels_x[0]'] = x[0].clone().detach()
        
        # Generate Logits for the Last Sequence in the Block Header
        x_drop = self.dropout(x)
        
        x_features = self.featureifier(x_drop)
        #logit_solution['logits'] = x_features['features'][:,:,:4].clone().detach().to(dtype=torch.float32)
        #logit_solution['nonce'] = torch.argmax(x_features['features'][:,:,:4], dim=1).to(dtype=torch.long)
        
        
        x_quantized = self.vq(x_features)
        logit_solution['dictionary_loss'] = x_quantized['dictionary_loss'].to(dtype=torch.float32)
        logit_solution['commitment_loss'] = x_quantized['commitment_loss'].to(dtype=torch.float32)
        
        
        x_reconstruct = self.reconstruction( x_features, x_quantized )
        
        header = x_reconstruct['reconstruction header']
        logit_solution['header'] = header.to(dtype=torch.float32)
        n_once =  header[:,-4:].clone().detach()
        reconstruct_n_once = (255*( (n_once+1)/2 ) ).to(dtype=torch.long)
        logit_solution['nonce'] = reconstruct_n_once
        logit_solution['logits'] = torch.nn.functional.one_hot(reconstruct_n_once, num_classes=256).to(dtype=torch.float32).permute(0,2,1)


        #outputs = self.classifier( x_features, labels=encoder_source )
        
        #header_in = encoder_source[0].unsqueeze(1)
        #header_reconst = x_reconstruct['reconstruction header'][0].unsqueeze(1)
        #header_img = torch.concat([header_in, header_reconst], dim=1)
        #logit_solution['pixels_outputs[0]'] = torch.repeat_interleave(header_img, 16, dim=0  )

        return logit_solution



############################################################################################################################################
#
#
############################################################################################################################################






# ==============================================================================================================================
#
#       EMBEDDING
#
# ==============================================================================================================================
class EmbeddingMethod(torch.nn.Module):
    def __init__(self, in_=1, out_=256, seq_len_=76, convbias_=True, depth_=None, embed_dim_=None):
        super().__init__()
        self.dtype=torch.float32
        self.device = get_device()
        self.drop_n_once_tokens = nn.Dropout(p=0.25)
        self.drop = nn.Dropout(p=0.00125)
        self.activate = nn.LeakyReLU()

        self.vocab_size=256
        self.embedding_dim = embed_dim_
        self.positional_enc = Positional_Encoding(self.vocab_size, self.embedding_dim)
        
        self.sqrt_dim_embed = np.sqrt(self.embedding_dim)
        self.input_seq_len = seq_len_

        self.starting_embedding_dim = 16
        self.first_block = nn.ModuleList( [ nn.Conv1d(1, 1, i_, stride=1, padding='same', padding_mode='reflect', bias=True, device=self.device) for i_ in range(1, self.starting_embedding_dim+1, 1) ] )
        
        self.quicknorm_1d = lambda x:  (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)


        self.spatial_correction = nn.Linear(self.input_seq_len, self.input_seq_len, device=self.device)

        self.residual_spatial_correction = ResNet( nn.Sequential(
                                                        nn.Linear(self.input_seq_len, self.input_seq_len//2, device=self.device),
                                                        self.activate,
                                                        nn.Linear(self.input_seq_len//2, self.input_seq_len, device=self.device),
                                                        self.activate ) )
    #
    #-------------------------------------------------------------------------------------
    def include_decoder_target_tokens(self, src: torch.Tensor, tgt=None) -> torch.Tensor:
        if tgt is None:
            tgt = torch.randint(0,255, (src.size(0), 4), device=src.device, dtype=src.dtype)
        return torch.concat( [src, tgt] , dim=1)
    #
    # ----------------------------------------------------------------------------
    def _pe(self, batches=1) -> torch.Tensor:
        n_batch = batches
        pe = self.positional_enc(self.input_seq_len).repeat(n_batch, 1, 1)
        pe.requires_grad_(requires_grad=False)
        pe = self.positional_enc(self.vocab_size).repeat(n_batch, 1, 1)
        pe = pe.permute(0,2,1)
        #pe = pe.transpose(1,-1)
        pe = pe[:,:self.input_seq_len,:self.vocab_size]
        return  pe
    #
    # -----------------------------------------------------------
    #@torch.autocast(device_type="cuda")
    def forward(self, src, tgt=None, train=None):
    
        src_start = self.include_decoder_target_tokens(src, tgt=(tgt if train else None))
        src_start[:, -4:] = self.drop_n_once_tokens(src_start[:, -4:].to(dtype=torch.float32))
        src_ = self.quicknorm_1d( src_start.to(dtype=torch.float32) ).unsqueeze(1)
        src_ = [ self.first_block[idx_](src_)  for idx_ in range(self.starting_embedding_dim)]
        #src_ = [ self.spatial_correction( self.drop(src_[idx_]) ) for idx_ in range(self.starting_embedding_dim)]
        src_ = [ self.residual_spatial_correction( self.drop(src_[idx_]) ) for idx_ in range(self.starting_embedding_dim)]
        src_ = torch.concat(src_, dim=1)
        return src_

    # ----------------------------------------------------------- # -----------------------------------------------------------
    # ----------------------------------------------------------- # -----------------------------------------------------------
        
        
        
        
        

# ==========================================================================================================================
# ==========================================================================================================================
#
#           CONV NET
#
# ==========================================================================================================================

class ConvNet(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.device = get_device()

        dim_features =     [   16,  32,   64,  128, 256]
        #conv_y_pool =      [ 80,  40,  20,   10,    5,    2 ]
        #conv_y_pool =      [ 76,  38,  19,   9, ]
        conv_y_pool =      [ 80,  40,  20,   10, ]

        conv_chans_in  = dim_features
        conv_chans_out = dim_features
        conv_last_layer_flags = [False, False, False, False]
        #conv_blocks = [3,3,3,3,3]
        conv_blocks = [3,3,3,3]
        conv_blocks = [2,2,2,2]
        #conv_blocks = [5,4,3,2]
        
        conv_chans = zip(conv_chans_in[:-1], conv_chans_out[1:], conv_y_pool, conv_blocks)

        self.n_blocks = len(conv_blocks)
        #self.mix_idx = ((3,0,1,2),(2,3,0,1),(1,2,3,0),(0,1,2,3))
  
        self.quicknorm = lambda x:  (x - torch.mean(x, dim=[1,2], keepdim=True)) / torch.std(x, dim=[1,2], keepdim=True)

                    
        self.blocks = nn.ModuleList( [ self.make_conv_block(in_,out_, seq_len=pool_, blocks=blocks_ ) for in_,out_,pool_,blocks_ in conv_chans] )
        
        self.pooling = [(nn.AdaptiveMaxPool1d(4) if flag_ else nn.MaxPool1d(2, stride=2, ceil_mode=False)) for flag_ in conv_last_layer_flags ]
    

        """
        self.spatial_max_pooling = nn.MaxPool1d(2, stride=2, ceil_mode=False)
        self.spatial_avg_pooling = nn.AvgPool1d(2, stride=2, ceil_mode=False)
        self.spatial_attn = nn.ModuleList( [nn.Sequential( nn.Linear(seq_len//2, seq_len//2, device=self.device),
                                            nn.LeakyReLU(),
                                            nn.Linear(seq_len//2, seq_len, device=self.device)) for seq_len in conv_y_pool])
        
        
        self.channel_max_pooling = nn.MaxPool1d(2, stride=2, ceil_mode=False)
        self.channel_avg_pooling = nn.AvgPool1d(2, stride=2, ceil_mode=False)
        self.channel_attn = nn.ModuleList( [ nn.Sequential(
                                                nn.Conv1d(out_, out_, 7, padding='same', padding_mode='reflect', device=self.device),
                                                nn.Sigmoid() ) for out_ in dim_features] )
        """


        self.activate = nn.LeakyReLU()
    #
    # -----------------------------------------------------------------------------
    def make_conv_block(self, input_channels, output_channels, seq_len=None, blocks=3, first_layer=False, final_layer=False):

        return nn.Sequential( *[ConvBlock(input_channels, input_channels, seq_len) for i_ in range(blocks-1)],
                                ConvBlock(input_channels, output_channels, seq_len)  )
    

    # -----------------------------
    def spatial_attention(self, x_in, idx=0):
        
        x_max = self.spatial_max_pooling(x_in)
        x_avg = self.spatial_avg_pooling(x_in)
        
        x_max =  self.spatial_attn[idx](x_max)
        x_avg =  self.spatial_attn[idx](x_avg)
        
        return x_in * torch.nn.functional.sigmoid(x_max + x_avg)
    # -----------------------------
    def channel_attention(self, x_in, idx=0):
        
        x_ = x_in.permute(0,2,1)
        x_max = self.channel_max_pooling(x_)
        x_avg = self.channel_avg_pooling(x_)
        
        x_ = torch.concat([x_max, x_avg], dim=-1)
        x_ = x_.permute(0,2,1)
        
        del x_max, x_avg
        return x_in * torch.nn.functional.sigmoid(self.channel_attn[idx](x_))
    #
    #
    # -----------------------------------------------------------------------------
    def forward(self, x):
        x_ = x
        
        unet_saved_states = [x_.clone()]
        
        for idx_ in range(self.n_blocks):
            x_in = x_
            x_out = self.blocks[idx_](x_in)
  
            #x_out = self.spatial_attention(x_out, idx=idx_)
            #x_out = self.channel_attention(x_out, idx=idx_)
            #x_out = x_in + x_out
            
            x_ = self.pooling[idx_](x_out)
            unet_saved_states.append(x_.clone())
 
        #x_lin = self.logit_linear( nn.Dropout(p=0.05)(x_).permute(0,2,1)).permute(0,2,1)
        #x_ = x_ + self.activate(x_lin)
        output_ = {'features':x_, 'unet_states':unet_saved_states}
        del x_, x_in, x_out, unet_saved_states
        return output_
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------



# ==========================================================================================================================
# ==========================================================================================================================
#
#           CONV BLOCKS
#
# ==========================================================================================================================
class ConvBlock(torch.nn.Module):
    def __init__(self, in_,out_,seq_len, groups=16, bottleneck=True, convbias=True):
        super().__init__()
        self.dtype=torch.float32
        self.device = get_device()
        #self.activate = nn.LeakyReLU()
        self.activate = torch.nn.Hardswish()
        #self.drop = nn.Dropout(p=0.0025)
        self.drop = nn.Dropout(p=0.0000125)
        self.pre_normalize = nn.LayerNorm( [in_, seq_len] , eps=1e-07, elementwise_affine=True)
        self.shortcut_conv = nn.Sequential( nn.Conv1d(in_, out_, 1, padding=0, padding_mode='zeros', bias=convbias, device=self.device),
                                                self.get_normalization(out_, seq_len=seq_len) )

        if bottleneck:
            #dim_bottleneck = out_ // 4
            dim_bottleneck = out_
            self.block  = nn.Sequential( *[
                nn.Conv1d( in_, dim_bottleneck, 1, padding='same', padding_mode='zeros', bias=convbias, device=self.device),
                self.get_normalization(dim_bottleneck, seq_len=seq_len),
                self.activate,
                nn.Conv1d(dim_bottleneck, dim_bottleneck, 3, padding='same', padding_mode='reflect', bias=convbias, device=self.device, groups=dim_bottleneck ),
                self.get_normalization(dim_bottleneck, seq_len=seq_len),
                self.activate,
                nn.Conv1d(dim_bottleneck, out_, 1, padding='same', padding_mode='zeros', bias=convbias, device=self.device),
                self.get_normalization(out_, seq_len=seq_len),
                ])

        if not bottleneck:
            self.block  = nn.Sequential( *[
                nn.Conv1d( in_, out_, 3, padding='same', padding_mode='zeros', bias=convbias, device=self.device),
                self.get_normalization(out_, seq_len=seq_len),
                self.activate,
                nn.Conv1d(out_, out_, 3, padding='same', padding_mode='zeros', bias=convbias, device=self.device),
                self.get_normalization(out_, seq_len=seq_len),
                ])


    #
    #-----------------------------------------------------------------------------
    def get_normalization(self, channels, seq_len=None):
        if seq_len is None:
            return nn.BatchNorm1d(channels, eps=1e-07)
        elif seq_len is not None:
            return nn.LayerNorm( [channels, seq_len], eps=1e-07, elementwise_affine=True)
    

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #@torch.autocast(device_type="cuda")
    def forward(self, x_in):
        x_in = self.drop( self.pre_normalize(x_in))
        return self.activate( self.shortcut_conv(x_in) +  self.block(x_in) )
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    
    








# ==========================================================================================================================
# ==========================================================================================================================
#
#           UPSCALE NET
#
# ==========================================================================================================================
class ReconstructionNet(torch.nn.Module):
    """
    # L_out = lambda L_in : (L_in -1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    #d = {'stride':2, 'padding':0, 'dilation':1, 'kernel_size':3, 'output_padding':0} (76)
    #d = {'stride':2, 'padding':1, 'dilation':1, 'kernel_size':4, 'output_padding':0} (80)
    #d = {'stride':self.stride, 'padding':self.padding, 'dilation':self.dilation, 'kernel_size':self.kernel_size, 'output_padding':self.output_padding}
    """
    def __init__(self,):
        super().__init__()
        self.dtype=torch.float32
        self.device = get_device()
        self.drop = nn.Dropout(p=0.00125)
        self.activate = nn.LeakyReLU()
        #conv_y_pool = [ 76, 38,  19,   9,     4 ]
        conv_y_pool = [80,  40,  20,   10,     5 ]
        conv_y_pool.reverse()
        self.x_spaces = conv_y_pool[1:]
        dim_features =     [ 8, 16,  32,   64,  128, 256 ]
        dim_features.reverse()
        self.layer_norm_dims = zip(dim_features[1:], self.x_spaces)
        self.stride = 2
        self.padding = 2
        dilation = 1
        self.kernel_size = 6
        self.output_padding = 0

        
        
        conv_chans_in  = [ 2*features for features in dim_features]
        conv_chans_out = [ features for features in dim_features]
        conv_blocks = [3,3,3,3]
        conv_blocks = [2,2,2,2]
        #conv_blocks = [2,3,4,5]
        
        conv_chans = zip(conv_chans_in, conv_chans_out, conv_y_pool, conv_blocks)
        

        conv_t_chans_in  = [ features for features in dim_features]
        conv_t_chans_out = [ features for features in dim_features[1:]]
        #conv_t_padding = [ 0,0,1,1,1 ]
        conv_transpose_chans = zip(conv_t_chans_in, conv_t_chans_out)

        conv_transpose_chans_A = zip(conv_t_chans_in, conv_t_chans_out)
        conv_transpose_chans_B = zip(conv_t_chans_in, conv_t_chans_out)
        conv_transpose_chans_C = zip(conv_t_chans_in, conv_t_chans_out)
        conv_transpose_chans_D = zip(conv_t_chans_in, conv_t_chans_out)
        conv_transpose_chans_E = zip(conv_t_chans_in, conv_t_chans_out)
        conv_transpose_chans_F = zip(conv_t_chans_in, conv_t_chans_out)


        self.resblocks = nn.ModuleList( [ self.make_conv_block(in_,out_, seq_len=pool_, blocks=blocks_ ) for in_,out_,pool_,blocks_ in conv_chans] )
        self.n_upscales = len(self.resblocks)
        self.upscale_norm = nn.ModuleList( [nn.LayerNorm( [c_, sq_], eps=1e-07, elementwise_affine=True) for c_,sq_ in self.layer_norm_dims])
        self.final_conv = nn.Conv1d(conv_t_chans_out[-2], 1, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device)
        self.quantized_output_correction = nn.Linear(80,80, device=self.device)

        self.quantized_output_correction = ResNet( nn.Sequential(
                                                        nn.Linear(80, 80//2, device=self.device),
                                                        self.activate,
                                                        nn.Linear(80//2, 80, device=self.device),
                                                        self.activate ) )
        # {'stride':2, 'padding':1, 'dilation':1, 'kernel':4, 'out_pad':0}      ()
        # {'stride':2, 'padding':2, 'dilation':1, 'kernel':5, 'out_pad':1}  👍  (B)
        # {'stride':2, 'padding':1, 'dilation':1, 'kernel':3, 'out_pad':1}  👍  (A)
        # {'stride':2, 'padding':1, 'dilation':2, 'kernel':2, 'out_pad':1}  ?   (D)
        
        
        """
        self.spatial_upscale = nn.ModuleList( [ torch.nn.Sequential( torch.nn.ConvTranspose1d(in_, out_, self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                        nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) ) for in_,out_ in conv_transpose_chans] )
        """
        self.spatial_multiscale_upscale_A =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 3, stride=2, padding=1, output_padding=1, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_A])

        """
        self.spatial_multiscale_upscale_B =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 5, stride=2, padding=2, output_padding=1, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_B])
        """
        
        self.spatial_multiscale_upscale_C =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 6, stride=2, padding=2, output_padding=0, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_C])

        """
        self.spatial_multiscale_upscale_D =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 2, stride=2, padding=1, output_padding=1, groups=out_, bias=True, dilation=2, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_D])
        """

        """
        self.spatial_multiscale_upscale_E =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 4, stride=2, padding=1, output_padding=0, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_E])
        """
        


        self.spatial_multiscale_upscale_F =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 9, stride=2, padding=4, output_padding=1, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_F])



        """ """
        # {'stride':2, 'padding':1, 'dilation':1, 'kernel':4, 'out_pad':0}


     
        """ # XXX
        self.spatial_multiscale_upscale_C =  nn.ModuleList( [
                torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(in_, out_, 3, stride=self.stride, padding=2, output_padding=3, groups=out_, bias=True, dilation=1, padding_mode='zeros', device=self.device, dtype=self.dtype),
                    nn.Conv1d(out_, out_, 1, stride=1, padding='same', padding_mode='zeros', bias=True, device=self.device) )
                    for in_,out_ in conv_transpose_chans_C])

        """



    # -----------------------------------------------------------------------------
    def make_conv_block(self, input_channels, output_channels, seq_len=None, blocks=3, first_layer=False, final_layer=False):

        return nn.Sequential( *[ConvBlock(input_channels, input_channels, seq_len) for i_ in range(blocks-1)],
                                ConvBlock(input_channels, output_channels, seq_len) )
    
    
    # ------------------------------------------------------------------------------------------
    def forward(self, encoder_output_dict: dict, quantizer_output_dict: dict) -> torch.Tensor:
        
        x_ = quantizer_output_dict['quantized_features']
        y_prev = encoder_output_dict['unet_states']
        y_prev.reverse()
        
        for n_ in range(self.n_upscales):

            x_ = torch.concat([x_, y_prev[n_]], dim=1)
            x_ = self.resblocks[n_](x_)

            
            #x_A = self.spatial_upscale[n_](x_)
            #x_D = self.spatial_multiscale_upscale_D[n_](x_)
            #x_ = x_A + x_D
            

            x_C = self.spatial_multiscale_upscale_C[n_](x_)
            x_A = self.spatial_multiscale_upscale_A[n_](x_)
            x_F = self.spatial_multiscale_upscale_F[n_](x_)
            x_ = x_C + x_A + x_F
            
            x_ = self.upscale_norm[n_](x_)
            x_ = self.activate(x_)
            
        x_final = self.final_conv(x_)
        x_final = self.quantized_output_correction( self.drop(x_final) )
        x_final = torch.nn.functional.hardtanh(x_final)
        
        output_ = {'reconstruction header': x_final.squeeze(1)}
        del x_, y_prev, x_final
        return output_
        
        

    



# =====================================================================================================================
#
#           CLASSIFIER
#
# =====================================================================================================================
class LinearClassifier(torch.nn.Module):
    def __init__(self,vocab_size=256, output_seq_len=4, input_seq_len=76):
        super().__init__()
        self.dtype=torch.float32
        self.device = get_device()
        self.activate = nn.LeakyReLU()
        self.sm_activate = nn.Softmax(dim=-1)
        #self.drop = nn.Dropout(p=0.35)
        self.drop = nn.Dropout(p=0.50)
        
        self.vocab_size = vocab_size
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.reduction_dim = self.vocab_size//4
        self.quicknorm = lambda x:  (x - torch.mean(x, dim=[1,2], keepdim=True)) / torch.std(x, dim=[1,2], keepdim=True)

        self.label_ff = self._linear_feedforward_for_skip()
                            
        self.skip_connection_0 = nn.Sequential( nn.Conv1d(1, 1, 1, device=self.device),
                                                nn.BatchNorm1d(1, device=self.device))
        
        self.vocab_head = nn.Linear(self.input_seq_len, 4*self.vocab_size, device=self.device)

    
        self.norm_stack = nn.BatchNorm1d(self.output_seq_len, device=self.device)
        
        self.double_down = nn.Linear(self.vocab_size, self.vocab_size, device=self.device)
        self.double_double_down = nn.Linear(self.vocab_size, self.vocab_size, device=self.device)

        self.mix_idx = ((0,1,2,3),(1,0,2,3),(2,0,1,3),(3,0,1,2))
        self.attn = nn.ModuleList( [torch.nn.MultiheadAttention(256, 4, dropout=0.50, bias=True, batch_first=True, device=self.device) for i in range(4)])
        
        self.atten_weight_0 = torch.nn.parameter.Parameter(data=torch.tensor(1.0, device=self.device))
        self.atten_weight_1 = torch.nn.parameter.Parameter(data=torch.tensor(1.0, device=self.device))
        self.atten_weight_2 = torch.nn.parameter.Parameter(data=torch.tensor(1.0, device=self.device))
        self.atten_weight_3 = torch.nn.parameter.Parameter(data=torch.tensor(1.0, device=self.device))

 
    #
    #-----------------------------------------------------------------------------------------------
    def _linear_feedforward_for_skip(self):
        return nn.Sequential( self.drop,
                            nn.Linear(self.input_seq_len, self.vocab_size, device=self.device),
                            nn.LeakyReLU(),
                            self.drop,
                            nn.Linear(self.vocab_size, self.vocab_size//2, device=self.device),
                            nn.LeakyReLU(),
                            self.drop,
                            nn.Linear(self.vocab_size//2, self.vocab_size//4, device=self.device),
                            nn.LeakyReLU(),
                            self.drop,
                            nn.Linear(self.vocab_size//4, self.vocab_size//2, device=self.device),
                            nn.LeakyReLU(),
                            self.drop,
                            nn.Linear(self.vocab_size//2, self.vocab_size, device=self.device),
                            nn.LeakyReLU(),
                            self.drop,
                            nn.Linear(self.vocab_size, self.input_seq_len, device=self.device)
                            )
    #
    #-----------------------------------------------------------------------------
    def _feedforward_embed(self, batched_1d_labels: torch.Tensor) -> list[torch.Tensor] :
        
        emb_ff = batched_1d_labels.to(dtype=torch.float32).unsqueeze(1)
        
        
        x_ff    = self.label_ff( emb_ff )
        x_skip  = self.skip_connection_0( emb_ff )
        x_comb  = self.activate( x_ff + x_skip )
        
        
        x_head = self.vocab_head( self.drop( x_comb ) )
        x_head = x_head.reshape(-1, self.output_seq_len, self.vocab_size)
        return x_head

    # ----------------------------------------------------------------------
    def feature_attn(self, x_):
        #  Logit Feature:  Xa - Mean(Xb,Xc,Xd)
        #x_features = torch.stack( [ x_[:,:,a_] * ((x_[:,:,b_]+x_[:,:,c_]+x_[:,:,d_])/3) for a_,b_,c_,d_ in self.mix_idx] , dim=1)
        p_ = [self.atten_weight_0, self.atten_weight_1, self.atten_weight_2, self.atten_weight_3]
        #x_features = torch.stack( [ x_[:,:,a_] + p_[a_]*self.attn[a_](x_[:,:,b_].unsqueeze(1), x_[:,:,c_].unsqueeze(1), x_[:,:,d_].unsqueeze(1))[0].squeeze(1) for a_,b_,c_,d_ in self.mix_idx] , dim=1)
        x_features = torch.stack([
                        x_[:,:,a_] + p_[a_] * self.attn[a_]((x_[:,:,a_] * x_[:,:,b_]).unsqueeze(1),
                                                            (x_[:,:,a_] * x_[:,:,c_]).unsqueeze(1),
                                                            (x_[:,:,a_] * x_[:,:,d_]).unsqueeze(1))[0].squeeze(1) for a_,b_,c_,d_ in self.mix_idx] , dim=1)
        #return self.batchnorm( x_features )
        return  x_features
    # ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    #@torch.autocast(device_type="cuda")
    def forward(self, x: torch.Tensor, labels=None) -> torch.Tensor:
        
        #x_A = x.reshape(-1, self.output_seq_len, self.vocab_size)
        #x_A = x
        #x_B = self._feedforward_embed(labels)
        #x_C = x_A + x_B
        #x_D = self.norm_stack( x_C )
        #x_D = self.sm_activate( x_D )
        
        #x_D = x
        #x_E = self.double_down( self.drop( x_D ) )
        #x_E = self.activate( x_E )
        
        #output = self.double_double_down( self.drop( x_E ) )
        x_features = self.feature_attn( self.drop(x) ) 
        
        output = self.double_double_down( self.drop( x_features ) )
        return output.permute(0,2,1)
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    
    
    
    


# =================================================================
# =================================================================
# =================================================================
# =================================================================
class ResNet(torch.nn.Module):
    def __init__(self, module):
        super(ResNet, self).__init__()
        self.SkipConnection = module

    # --------------------------------
    #@torch.autocast(device_type="cuda")
    def forward(self, inputs):
        return inputs + self.SkipConnection(inputs)






# =================================================================
# =================================================================
# =================================================================
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
        e_i_ts = torch.ones( (embedding_dim, num_embeddings,), device=self.device, dtype=torch.float32)#.uniform_(-limit, limit)
        self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))
        nn.init.xavier_uniform_(e_i_ts, gain=nn.init.calculate_gain('relu'))


    # ----------------------------------------------------------------------
    def forward(self, x_in):
        #flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        
        x = x_in['features'].clone()
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
            encoding_indices.view(x.shape[0], x.shape[1]), self.e_i_ts.transpose(0, 1)
        )#.permute(0, 2, 1)

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
