import numpy as np
import torch
import torch.nn as nn
import gc
import os, os.path
import pytorch_lightning as pl
from copy import deepcopy
from dataclasses import dataclass, asdict
from pytorch_lightning import loggers as pl_loggers

from .mask import Masks

from .deep_decoder import DeepDecoder, DecoderModelConfig
from .basic_net import Basic_Net

from .n_once_stats import NonceDigitStats
#from .positional_encoding import Positional_Encoding
from .randomized_preblock_header import Randomized_Preblock_Header
from ..framework.tokenizer import Tokenizer
#from ..framework.tokenizer_single_digit import Tokenizer
from ..framework.labeler import Labeler
from ..framework.functions import LogitRankingsFromKnownSolutions
from ..framework.functions import nonce_token_to_int, check_hashes_from_generated_nonce
from ..framework.functions import logit_classes_above_random_chance
from ..data_io.show import  PlotLogitClassProbability, pixelize
from ..data_io.utils import save_json_file, load_json_file, set_device_params
#from ..data_io.utils import check_if_time_is_outside_of_low_cost_energy_zone
from ..data_io.const import CHECKPOINT_FILE_PATH, DATA_PATH






#----------------------------------------------------------------
#                       '📏'
#----------------------------------------------------------------
class TransformerDecoder(pl.LightningModule):
    """
    Transformer Decoder utilizes the output from a Transformer Encoder to generate a single logit.
    During inference, the Decoder will generate the full 4 Logits which represent the Nonce value.
    """
    def __init__(self, n_epoch=1, n_data_batches=1, checkpoint=None, learning_rate=0.01):
        super().__init__()

        set_device_params()

        # -----------  Model Parameters -----------
        self.model_type = 'Lightning Decoder'
        self.checkpoint = checkpoint
        self.encoder_input_dim = 80
        self.decoder_input_dim = 4
        self.rnd_preblock = Randomized_Preblock_Header()
        self.tokenizr = Tokenizer()
        self.labelr = Labeler()

        # ----------- Decoder Module Parameters ---------
        mdl_config = DecoderModelConfig()
        self.mask = Masks(num_heads=mdl_config.num_heads)
        self.deep_model = DeepDecoder( **asdict(DecoderModelConfig()) )
        #self.deep_model = Basic_Net( )
        self.deep_model.requires_grad_(requires_grad=True)

        # --------- Optimizer Parameters --------
        self.n_data_batches = n_data_batches
        self.max_epochs = n_epoch
        steps_to_half_max_epoch = int(0.5*self.max_epochs*self.n_data_batches)
        self.warmup_steps = (steps_to_half_max_epoch if steps_to_half_max_epoch < 60000 else 60000)
        self.regularize = False
        self.max_learning_rate = learning_rate #3.9e-3 #1.17e-2
        self.lambda_L1 = self.max_learning_rate / (4*self.max_epochs)
        self.decay_rate = self.max_learning_rate / (4*self.max_epochs)

    
        # --------- Loss Function Parameters -------
        self.val_test_every = 4
        self.save_every_epoch = False
        smoothed_by = np.array([1., 1., 1., 1.]) * 0.0
        self.validation_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.0)
        self.loss_fn_per_epoch = nn.CrossEntropyLoss(reduction='sum', label_smoothing=0.0)
        self.loss_weights = self.load_weights()
        if self.loss_weights is not None:
            if self.loss_weights.size(0) == 256:
                self.loss_0 = nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothed_by[0], weight=self.loss_weights)
                self.weighted_train_loss_fn = [self.loss_0, self.loss_0, self.loss_0, self.loss_0]
            elif self.loss_weights.size(0) == 4:
                #self.loss_weights = torch.sqrt(self.loss_weights)
                self.loss_0 = nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothed_by[0], weight=self.loss_weights[0])
                self.loss_1 = nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothed_by[1], weight=self.loss_weights[1])
                self.loss_2 = nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothed_by[2], weight=self.loss_weights[2])
                self.loss_3 = nn.CrossEntropyLoss(reduction='sum', label_smoothing=smoothed_by[3], weight=self.loss_weights[3])
                #self.weighted_train_loss_fn = [self.loss_0, self.loss_1, self.loss_2, self.loss_3]
                #self.weighted_train_loss_fn = [self.hash_loss_fn, self.hash_loss_fn, self.hash_loss_fn, self.hash_loss_fn]
                self.weighted_train_loss_fn = [self.loss_fn_per_epoch] * self.decoder_input_dim
                del self.loss_weights
        else:
            self.weighted_train_loss_fn = [self.validation_loss_fn, self.validation_loss_fn, self.validation_loss_fn, self.validation_loss_fn]

        # --------- Recoding Validation Metrics --------
        self.first_training_logit = None
        self.first_test_logit = None
        self.extra_n_once = False
        self.extra_n_once_trials = 2
        self.nonce_digit_stats = NonceDigitStats(device=self.device, n_digits=self.decoder_input_dim).to(device=self.device)
        self.training_classes_above_random = {}
        self.total_validation_preblocks_accepted = 0
        self.total_randomized_preblocks_accepted = 0

        self.save_hyperparameters()
        #self.vae_mse_loss = nn.MSELoss()
        self.vae_mse_loss = torch.nn.HuberLoss()
        self.quicknorm_0d = lambda x:  ((x - torch.mean(x.to(dtype=torch.float32), dim=0, keepdim=True)) / torch.std(x.to(dtype=torch.float32), dim=0, keepdim=True))
        self.train_decoder_hash = False
        self.train_decoder_n_once = True
        self.n_block_tail = 4
        self.augment = False

    #
    # ###################################################################################### #
    # ------------------------------      Initialize     ----------------------------------- #
    # ------------------------------                     ----------------------------------- #
    # ###################################################################################### #
    #
    # ====================================================================
    def setup(self, stage) -> None:
        # Get tensorboard logger
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                self.tb_logger = logger.experiment
                break
    #
    # ====================================================================
    def load_weights(self) -> torch.Tensor:
        try:
            print(f"[ ] .... Loss Weights in: {DATA_PATH + f'training_weights_for_loss.json'}")
            weights = load_json_file(DATA_PATH + f'training_weights_for_loss.json')
            print(f"[✔] .... Loaded.")
            return torch.tensor(np.array(weights), dtype=torch.float32, device=self.device)
        except:
            print(f"[✘] ....  Training Weights Failed to Load Properly.  [✘]")
            exit()
    #
    # ====================================================================
    @torch.no_grad()
    def _initialize_tensors(self) -> None:
        
        for params in self.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.normal_(params, mean=0.0, std=0.01)

            if isinstance(params, nn.LayerNorm) | isinstance(params, nn.BatchNorm1d) | isinstance(params, nn.InstanceNorm1d):
                nn.init.normal_(params.weight, mean=1.0, std=0.001)
                nn.init.normal_(params.bias, mean=0.0, std=0.001)
                
      

        """
        ## Hash Encoder Model:
        self._init_from_file( filename = os.getcwd()+f'/checkpoints/Block-and-Hash-Encoder/Hash-encoder/ver_256/QuickBitsNN_DeepDecoder-encoder_(Deep_Decoder_cuda_mdl.1300ver.0ckp)_008ep.pth',
            model_instance = self.deep_model.hash_encoder_model,
            trainable = False)
        
        ## Hash Encoder Embedding:
        self._init_from_file( filename = os.getcwd()+f'/checkpoints/Block-and-Hash-Encoder/Hash-encoder_embedding/ver_256/QuickBitsNN_DeepDecoder-encoder_embedding_(Deep_Decoder_cuda_mdl.1300ver.0ckp)_008ep.pth',
            model_instance = self.deep_model.hash_encoder_embedding,
            trainable = False, )
        """
        return
    #
    # ========================================================================================
    def _init_from_file(self, filename=f'', model_instance=None, trainable=True) -> None:
        model_instance = self._load_saved_model(filename, model_instance)
        model_instance.requires_grad_(requires_grad=trainable)
        return

    # ========================================================================================
    def _init_embedding_from_file(self, filename=f'', trainable=True) -> None:
        self.deep_model.embedding = self._load_saved_model(filename, self.deep_model.embedding)
        self.deep_model.requires_grad_(requires_grad=trainable)
        return
    #
    # ========================================================================================
    def _load_saved_model(self, model_path_name, model_instance):
        #file_name_path_model_state_dict = CHECKPOINT_FILE_PATH + f'{model_path_name}.pth'
        file_name_path_model_state_dict = model_path_name
        try:
            print(f"[⚙️ ]  Importing Model From File ....")
            print(f"\t\"{model_path_name}\"")
            model_instance.load_state_dict(torch.load(file_name_path_model_state_dict, map_location=self.device), strict=False)
            print(f"[✔︎] ....  was Successfully Loaded! [✔︎]")
        except:
            print(f"[✘] ....  has Failed to Load. [✘]")
            exit()

        return model_instance
    #
    # ========================================================================================
    def _save_model_trained(self, model, type=f'', quiet=True) -> None:
        if not quiet:
            print(f"[🚉 ]  Saving Model Checkpoint as State Dictionary ....")
        save_epoch = str(self.trainer.current_epoch).zfill(3)
        file_name_path_model_state_dict = f'{self.checkpoint.checkpoint_path_to_save_file(type=type)}_{save_epoch}ep'
        torch.save(model.state_dict(), file_name_path_model_state_dict + f'.pth')
        if not quiet:
            self.print(f"[✔︎] .... Model State Saved as: \"{file_name_path_model_state_dict}\"\n")
        return

    # ====================================================================
    def step_check(self) -> bool:
        """ Select spacing for L1 loss to be applied. """
        # epoch_ = (1+self.trainer.current_epoch) % 2 == 0
        epoch_ = (self.trainer.current_epoch) >= 0
        epoch_2 = (1 + self.trainer.current_epoch) % self.val_test_every == 0
        step_ = (self.trainer.global_step % 1) == 0
        return ((epoch_ & step_) & epoch_2) | self.save_every_epoch


    #
    # ###################################################################################### #
    # ------------------------------      Forward        ----------------------------------- #
    # ------------------------------                     ----------------------------------- #
    # ###################################################################################### #
    def forward(self, src, tgt, train=False, masked_digit=3) -> torch.Tensor:
        return self.deep_model(src, tgt, train=train, masked_digit=masked_digit)

   #
    # ====================================================================
    def on_train_epoch_start(self) -> None:
        if (self.first_test_logit is not None) & self.step_check():
            self._plot_updated_item(train=True)
        return

    #
    # ====================================================================
    def on_train_epoch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.deep_model.dropout.p > 1.e-5:
            new_p = self.deep_model.dropout.p * (0.666)
            self.deep_model.dropout.p = new_p
            self.deep_model._update_dropout(new_p)
            self.print(f"Dropout Updated to: p = {self.deep_model.dropout.p} on epoch end: {self.trainer.current_epoch}")
        elif 1.e-8 < self.deep_model.dropout.p < 1.e-5 :
            new_p = 0.00
            self.deep_model.dropout.p = new_p
            self.deep_model._update_dropout(new_p)
            self.print(f"Dropout Stopped with: p = {self.deep_model.dropout.p} on epoch end: {self.trainer.current_epoch}")
        else:
            pass
        
        
        
        #if check_if_time_is_outside_of_low_cost_energy_zone():
        #    exit()
        return


    # -----------------------------------------------------------------------------------------
    #
    # ###################################################################################### #
    # ------------------------------        Train        ----------------------------------- #
    # ###################################################################################### #
    # ========================================      # ========================================
    # ========================================      # ========================================
    def training_step(self, batch, batch_idx) -> dict:


        (block_header, block_hash) = batch
        if self.train_decoder_hash & (not self.train_decoder_n_once):
            decoder_labels = block_hash
            if self.augment:
                (block_header, decoder_labels) = self.augment_data_during_training(block_header, block_hash)
        else:
            decoder_labels = block_header[:, -self.n_block_tail:]

        #if (self.first_test_logit is None):
        #    self.first_test_logit = [block_header[:1], decoder_labels[:1]]

        batch_ = block_header.size(0)
        logit_solution = self(block_header, decoder_labels, train=True)
        full_losses = self.combined_loss( logit_solution, decoder_labels )
            
        self.log('train_loss/step/', full_losses['loss'], batch_size=batch_, on_step=True, on_epoch=False)
        #self.log_dict({'train_loss/epoch/': full_losses['loss'], 'step':self.current_epoch+1 }, batch_size=batch_, on_step=False, on_epoch=True)
        del logit_solution
        return full_losses
        
    #
    # ====================================================================
    # ====================================================================
    def combined_loss(self, logit_solution, labels ) -> dict:

        #  Cross Entropy
        loss_total = self.cross_entropy_loss(logit_solution, labels)

        #  Regularization Loss
        if self.step_check():
            if self.regularize:
                loss_total = loss_total + self.L1_regularization()

        # Vector Quantization of Latent Space Loss 
        #vae_loss = self.VQ_VAE_loss(logit_solution, labels)
        #loss_total = loss_total + vae_loss

        loss = {'loss': loss_total}
        return loss


    #
    # ====================================================================
    def cross_entropy_loss(self, logit_solution, labels):
        logits = logit_solution['logits']
        items_per_batch, vocab_len, n_digits = logits.shape

        logit_digit_loss = [(self.weighted_train_loss_fn[n_](logits.clone()[:, :, n_], labels[:, n_]))/items_per_batch for n_ in range(n_digits)]
        _ = [self.log(f'train_loss/t{n_}', logit_digit_loss[n_]) for n_ in range(self.decoder_input_dim)]

        ls = self.validation_loss_fn(logits, labels)
        self.log('train_loss/', ls, on_step=False, on_epoch=True)


        loss_total = self.validation_loss_fn(logits, labels)

        # remove 'solved' digits from loss function
        #min_loss_threshold = 0.01
        #n_solved = int(sum([(1 if x_ < min_loss_threshold else 0) for x_ in logit_digit_loss]))

        # Arithmatic Mean
        #loss_total = sum(logit_digit_loss) / (n_digits - n_solved)
        
        # Geometric Mean
        #loss_total = torch.pow(torch.prod(torch.tensor(logit_digit_loss)), (1/n_digits))
        
        # Harmonic Mean
        #loss_total = n_digits / sum([1/x for x in logit_digit_loss])

        return loss_total
    

    #
    # ====================================================================
    def L1_regularization(self) -> torch.Tensor:
        l1_reg = torch.tensor(0., requires_grad=True, device=self.device)

        for name, param in self.deep_model.named_parameters():
            #if 'weight' in name:
            if (param.dim() < 3 )& (param.dim() != 0):
                l1_reg = l1_reg + torch.linalg.norm(param, 1)

        l1_reg = self.lambda_L1 * l1_reg
        self.log(f'L1_loss_(param)', l1_reg)
        return l1_reg


    #
    # ============================================================================
    def VQ_VAE_loss(self, logit_solution, labels, beta=0.25) -> float:
    
        vae_loss = beta * logit_solution['commitment_loss']
        vae_loss += logit_solution['dictionary_loss']
        self.log('vae_loss/', vae_loss, on_step=True, on_epoch=True)
        return vae_loss


    #
    # ###################################################################################### #
    # ------------------------------      Validate       ----------------------------------- #
    # ###################################################################################### #
    # ===================================
    def validation_step(self, batch, batch_idx) -> dict:

        (block_header, block_hash) = batch
        if self.train_decoder_hash & (not self.train_decoder_n_once):
            decoder_labels = block_hash
        else:
            decoder_labels = block_header[:, -self.n_block_tail:]

        #(block_header, block_tail) = batch
        #decoder_labels = block_tail[:,-self.decoder_input_dim:]
        
        if (self.first_test_logit is None) & (self.trainer.current_epoch > 0):
            self.first_test_logit = [block_header[:1], decoder_labels[:1]]
        
        logit_solution = self(block_header, decoder_labels, train=False)
        auto_logits = logit_solution['logits']
        candidate_nonce = logit_solution['nonce']

        batch_, vocab_, digits_ = auto_logits.shape
        single_val_losses = [self.validation_loss_fn(auto_logits[:, :, idx_], decoder_labels[:, idx_]) for idx_ in range(digits_)]
        _=[self.log(f'val_loss/v{idx_}/', single_val_losses[idx_], batch_size=batch_, on_step=True, on_epoch=True) for idx_ in range(digits_)]

        combined_val_loss = self.validation_loss_fn(auto_logits, decoder_labels)
        self.log(f'val_loss/raw', combined_val_loss, batch_size=batch_, on_step=True, on_epoch=True)

        avg_val_loss = digits_ / sum([1/x for x in single_val_losses])
        self.log(f'val_loss/h_mean/', avg_val_loss, batch_size=batch_, on_step=True, on_epoch=True)


        
        accuracy = self.nonce_digit_stats.logit_argmax_eqtokens_per_label_digit(auto_logits, decoder_labels)
        self.log('accuracy/', accuracy, batch_size=(batch_*self.decoder_input_dim), on_step=True, on_epoch=True)

        above_rand = logit_classes_above_random_chance(auto_logits, decoder_labels)
        self.training_classes_above_random['sum'] += above_rand.to(device=self.training_classes_above_random['sum'].device)
        self.training_classes_above_random['total'] += auto_logits.size(0)
        #del above_rand

        self.nonce_digit_stats.logit_known_class_probability(auto_logits, decoder_labels)
        self.nonce_digit_stats.accumulate_eqtokens_per_label_digit(candidate_nonce, decoder_labels)
        
        
        # Evaluate Block Acceptance with Generated n_Once
        if self.train_decoder_n_once:
            if self.extra_n_once:
                multi_n_once = self.select_solutions_from_logits(auto_logits, trials=self.extra_n_once_trials)
                multi_n_once = torch.concat([multi_n_once, candidate_nonce.unsqueeze(0) ], dim=0)
                n_acptd = sum([ self.test_nonce_preblock_acceptance(block_header[:, :-self.n_block_tail], n_once_,
                                          filename_ext=f'_known_preblocks_with_accepted_nonce.json',
                                          batch_idx=batch_idx) for n_once_ in multi_n_once.unbind()] )
            else:
                n_acptd = self.test_nonce_preblock_acceptance(block_header[:, :-self.n_block_tail] , candidate_nonce,
                                          filename_ext=f'_known_preblocks_with_accepted_nonce.json',
                                          batch_idx=batch_idx)
            if n_acptd > 0:
                self.log(f'Validation Headers Accepted/Current', n_acptd, on_step=True, on_epoch=False)
                self.total_validation_preblocks_accepted += n_acptd
                self.log(f'Validation Headers Accepted/Total', self.total_validation_preblocks_accepted, on_step=True, on_epoch=False)
                if self.trainer.current_epoch != 0:
                    self.log(f'Acceptance Rate/Validation', ( self.total_validation_preblocks_accepted/self.trainer.current_epoch) )
                
        return {'val_loss': combined_val_loss}

    # =======================================================================
    def on_validation_epoch_start(self) -> None:
        self.training_classes_above_random = {'sum': 0, 'total': 0}
        self.training_classes_above_random['sum'] = torch.zeros((self.decoder_input_dim,), dtype=torch.long, device=self.device)
        self.training_classes_above_random['total'] = torch.zeros((1,), dtype=torch.long, device=self.device)


        if ((self.trainer.current_epoch + 1) == self.max_epochs) | self.save_every_epoch:
            
            self._save_model_trained(self.deep_model.block_encoder_model, type=self.deep_model.model_type + f'block-encoder', quiet=False)
            self._save_model_trained(self.deep_model.block_encoder_embedding, type=self.deep_model.model_type + 'block-encoder_embedding', quiet=False)
            
            #self._save_model_trained(self.deep_model.hash_encoder_model, type=self.deep_model.model_type + f'hash-encoder', quiet=False)
            #self._save_model_trained(self.deep_model.hash_encoder_embedding, type=self.deep_model.model_type + 'hash-encoder_embedding', quiet=False)
            
            self._save_model_trained(self.deep_model.decoder_embedding, type=self.deep_model.model_type + '-decoder_embedding', quiet=False)
            self._save_model_trained(self.deep_model.block_decoder_model, type=self.deep_model.model_type + f'-decoder', quiet=False)
            
            self._save_model_trained(self.deep_model, type=self.deep_model.model_type, quiet=False)

        """"""
        if self.train_decoder_n_once:
            for zeros_ in range(0,64,2):
                (rnd_block_header, rnd_block_tail) = self.rnd_preblock.fake_data(leading_zeros=zeros_, samples=32, from_class_pdf=True)
                rnd_header = torch.concat([rnd_block_header, rnd_block_tail], dim=-1)
                logit_solution = self(rnd_header, rnd_block_tail, train=False)
                
                rnd_candidate_nonces = logit_solution['nonce']
                if self.extra_n_once:
                    multi_n_once = self.select_solutions_from_logits(logit_solution['logits'], trials=self.extra_n_once_trials)
                    multi_n_once = torch.concat([multi_n_once, rnd_candidate_nonces.unsqueeze(0)], dim=0)
                    n_acptd = sum([ self.test_nonce_preblock_acceptance(rnd_block_header, n_once_,
                                        filename_ext=f'_randomized_preblocks_with_accepted_nonce.json',
                                        batch_idx=-1) for n_once_ in multi_n_once.unbind()] )
                else:
                    n_acptd = self.test_nonce_preblock_acceptance(rnd_block_header, rnd_candidate_nonces,
                                                                    filename_ext=f'_randomized_preblocks_with_accepted_nonce.json',
                                                                    batch_idx=-1)

                if n_acptd > 0:
                    self.log(f'accepted_random_preblocks_({zeros_}_leading_zeros)', n_acptd)
                    self.total_randomized_preblocks_accepted += n_acptd
                    self.log(f'total_randomized_preblocks_accepted', self.total_randomized_preblocks_accepted)
                    if self.trainer.current_epoch != 0:
                        self.log(f'acceptance rate/random', (self.total_randomized_preblocks_accepted/self.trainer.current_epoch) )
        """"""
        return

    # ====================================================================
    def on_validation_epoch_end(self) -> None:

        # Skip over stats from Sanity Check and First Training Epoch
        if self.trainer.current_epoch < 1:
            self.nonce_digit_stats.reset_stats()
            return



        # --- Save Stats of Nonce Digits
        tb_log = self.nonce_digit_stats.validation_epoch_end_average_eq_tokens_by_digit()
        self.tb_logger.add_scalars(f'eq_tokens_per_digit', tb_log, self.trainer.global_step)
        avg_eq_tokens = self.nonce_digit_stats.validation_epoch_end_eq_tokens()
        self.log('eq_tokens', avg_eq_tokens, batch_size=self.nonce_digit_stats.n_prob,  on_step=False, on_epoch=True)


        avg_total_prob_of_known_nonce = self.nonce_digit_stats.validation_epoch_end_prob_of_known_n_once()
        self.log('avg_total_prob_of_known_nonce', avg_total_prob_of_known_nonce)


        avg_total_logprob_of_n_once_tokens = self.nonce_digit_stats.validation_epoch_end_mean_logprob_of_n_once_tokens()
        self.log('avg_total_logprob_of_n_once_tokens', avg_total_logprob_of_n_once_tokens)

        
        prob_of_known_labels_log = self.nonce_digit_stats.validation_epoch_end_known_label_prob_per_digit()
        self.tb_logger.add_scalars(f'label_prob_per_digit', prob_of_known_labels_log, self.trainer.global_step)

        # --- Classes Above Random Chance
        avg_above_random = self.training_classes_above_random['sum'].to(dtype=torch.float32) / self.training_classes_above_random['total'].to(dtype=torch.float32)
        _=[self.log(f'percent_of_all_classes_with_probability_above_random_chance/d{i_}', avg_above_random[i_], on_epoch=True ) for i_ in range(avg_above_random.size(0))]
        self.log(f'percent_of_all_classes_with_probability_above_random_chance', avg_above_random.mean(dim=0), on_epoch=True )

        if (self.first_test_logit is not None):
            self._plot_updated_item(train=False)

        # --- Save Copy of Model
        if self.nonce_digit_stats.check_if_best_run(above_rand=avg_above_random.mean(dim=0).item()):
            self._save_model_trained(self.deep_model, type=self.deep_model.model_type, quiet=False)
            self._save_model_trained(self.deep_model.block_encoder_model, type=self.deep_model.model_type+f'block-encoder', quiet=False)
            self._save_model_trained(self.deep_model.block_encoder_embedding, type=self.deep_model.model_type+'block-encoder_embedding', quiet=False)


        # --- Reset for Next Epoch
        self.nonce_digit_stats.reset_stats()
        return

    # ###################################################################################### #
    # ------------------------------       Predict       ----------------------------------- #
    # ###################################################################################### #
    def predict_step(self, batch, batch_idx) -> dict[str, int]:
        """ One - Preblock Header will be generated, blocktail will be blank. """
        (time_index, block_header, block_tail) = batch
        logit_solution = self(block_header, block_tail, train=False)
        predicted_nonce = logit_solution['nonce']
        (time, nonce) = self.test_block_header_acceptance(time_index, block_header, predicted_nonce)
        return {'nonce': nonce, 'time': time}


    # ====================================================================
    def select_solutions_from_logits(self, logits: torch.Tensor, trials=1 ) -> torch.Tensor:
        """ NOTE: Solving Multinomial on Cuda is significantly SLOWER than on CPU!
                Transfer tensors to CPU before, then return them to device! (negligible time compared to solving)
        :param logits: raw lagits from the model output
        :param trials: number of different n_once to generate
        :return: tensor.size([trials, batch, vocab])
        """
        assert trials > 0
        soft_logits = logits.to(device='cpu')
        return torch.stack([
                    torch.stack([
                        torch.multinomial(soft_logits[b_, :, d_], num_samples=trials, replacement=True).to(dtype=torch.long).squeeze(-1)
                            for d_ in range(soft_logits.size(2))
                            ], dim=-1)
                        for b_ in range(soft_logits.size(0))
                    ], dim=1).to(device=self.device)

        """
        solns = torch.zeros((trials, logits.size(0), logits.size(2), ), dtype=torch.long, device=self.device)
        for b_ in range(logits.size(0)):
            for d_ in range(logits.size(2)):
                solns[:, b_, d_] = torch.multinomial(logits[b_, :, d_].softmax(0), num_samples=trials, replacement=True).to(
                    dtype=torch.long).squeeze(-1)
        return solns
        """

    # =====================================================================
    def test_block_header_acceptance(self, time_index, block_header, candidate_nonce) -> tuple:
        n_accepted = 0
        for idx_ in range(block_header.size(0)):
            accepted_nonces = check_hashes_from_generated_nonce(block_header[idx_], candidate_nonce[idx_])
            if accepted_nonces is not None:
                n_accepted += len(accepted_nonces)
                detkn_nonce = [nonce_token_to_int(n_) for n_ in accepted_nonces]
                return (time_index[idx_].item(), detkn_nonce[0])
        del accepted_nonces
        return (None, None)
    #
    # =====================================================================
    def test_nonce_preblock_acceptance(self, block_header, candidate_nonce,
                                       filename_ext=f'',
                                       batch_idx=0) -> int:
        n_accepted = 0
        items_per_batch, word_len = block_header.shape
        for idx_ in range(items_per_batch):
            accepted_nonces = check_hashes_from_generated_nonce( block_header[idx_], candidate_nonce[idx_] )
            if accepted_nonces is not None:
                n_accepted += len(accepted_nonces)
                filename=self.checkpoint.log_path()+filename_ext
                prev_data=None
                try:
                    prev_data = load_json_file(filename)
                except:
                    pass
                if prev_data is None: prev_data = {}
                preblock = self.tokenizr.detokenize(block_header[idx_])
                detkn_nonce = [nonce_token_to_int(n_) for n_ in accepted_nonces]
                if preblock in prev_data:
                    _=[prev_data[preblock].append(n_) for n_ in detkn_nonce if n_ not in prev_data[preblock]]
                else:
                    prev_data.setdefault(preblock, detkn_nonce)
                save_json_file(prev_data, filename)
                del prev_data
        return n_accepted


    # ###################################################################################### #
    #           PLOTS
    # ###################################################################################### #
    def plots_from_logit_solution(self, logit_solution:dict, train=False, block_header=None, nonce_labels=None, batch_idx=0 ) -> None:

        file_label = f'TRAIN' if train else f'VAL'

        # Embedding
        if not train:
            emb_pxl = logit_solution['pixels_x[0]']
            emb_pxl = torch.repeat_interleave(emb_pxl, 2, dim=-1 )
            pixelize(emb_pxl).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/init_emb_[0]_{self.trainer.current_epoch}.png' )
            pixelize(self.quicknorm_0d(emb_pxl)).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/qn(init_emb_[0])_{self.trainer.current_epoch}.png' )
 
        # Encoder Output
        enc_out = logit_solution['encoder_output']
        enc_out = torch.repeat_interleave(enc_out, 2, dim=-1 )
        pixelize(enc_out).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_encoder_output_{self.trainer.current_epoch}.png' )
        pixelize(self.quicknorm_0d(enc_out)).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_qn(encoder_output)_{self.trainer.current_epoch}.png' )
        
        # Decoder Output
        dc_out = logit_solution['decoder_output']
        dc_out = torch.repeat_interleave(dc_out, 8, dim=-1 )
        pixelize(dc_out).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_decoder_output_{self.trainer.current_epoch}.png' )
        pixelize(self.quicknorm_0d(dc_out)).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_qn(decoder_output)_{self.trainer.current_epoch}.png' )

        # Decoder Logits
        lgt_prb = logit_solution['logits'][0].softmax(0)
        lgt_prb = torch.repeat_interleave(lgt_prb, 4, dim=-1 )
        pixelize(lgt_prb).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_decoder_logits_{self.trainer.current_epoch}.png' )


        # Reconstruction of N_once
        if self.train_decoder_n_once:
            #header_in = torch.concat([block_header[0], nonce_labels[0]], dim=-1)
            #header_in = header_in.to(dtype=torch.float32)
            #header_reconst = logit_solution['header'][0].to(dtype=torch.float32)
            #header_img = torch.stack([header_in, header_reconst], dim=0)
            #header_img = torch.repeat_interleave(header_img, 16, dim=0  )[:,-self.decoder_input_dim:]
            header_in = nonce_labels[0].to(dtype=torch.float32)
            header_reconst = logit_solution['nonce'][0].to(dtype=torch.float32)
            header_img = torch.stack([header_in, header_reconst], dim=0)
            header_img = torch.repeat_interleave(header_img, 3, dim=0  )
            
            min_max = ( [0,255] if self.decoder_input_dim == 4 else [0,15])
            pixelize(header_img, min_max=min_max).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_blocktail_reconstruction_{self.trainer.current_epoch}.png' )
        
        if self.train_decoder_hash:
            header_in = nonce_labels[0].to(dtype=torch.float32)
            header_reconst = logit_solution['nonce'][0].to(dtype=torch.float32)
            header_img = torch.stack([header_in, header_reconst], dim=0)
            header_img = torch.repeat_interleave(header_img, 3, dim=0  )
            min_max = [0,255]
            pixelize(header_img, min_max=min_max).save( CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx) + f'/({file_label})_hash_reconstruction_{self.trainer.current_epoch}.png' )
        
        return
        
    #
    # ====================================================================
    def _log_logit_plots(self, auto_logits, candidate_nonces, accepted_nonce, batch_idx, label=f'') -> None:
        f_path = CHECKPOINT_FILE_PATH + self.checkpoint.checkpoint_path_to_img(epoch=self.current_epoch, batch=batch_idx)
        #print(f"{auto_logits = }")
        #print(f"{auto_logits.shape = }")
        img = PlotLogitClassProbability(auto_logits,
                                        predicted=candidate_nonces[0, 0],
                                        solved=accepted_nonce,
                                        filepath=f_path,
                                        filename=f'[LogitProbability]_{label}_{self.checkpoint.checkpoint_name()}_{self.checkpoint.log_ver}logver_{batch_idx}batch_{str(self.global_step).zfill(7)}step.png',
                                        return_figure=True,
                                        prenormalized=False)
        self.tb_logger.add_figure(f"Logit_Class_Probability/{label}_{batch_idx}", img,
                                        global_step=self.global_step,
                                        close=True)
        del img
        return

    # ====================================================================
    @torch.no_grad()
    def _plot_updated_item(self, train=True):

        with torch.no_grad():
            # self.first_test_logit = [block_header[:1], decoder_labels[:1]]
            solution = self(self.first_test_logit[0], self.first_test_logit[1], train=train)

        if (not self.train_decoder_hash):
            solution_logit = solution['logits']
            solution_nonce = solution['nonce']
            self.first_training_logit = [solution_logit[:1], solution_nonce[:1].unsqueeze(0),
                                         self.first_test_logit[1][0], 0]
            self._log_logit_plots(*self.first_training_logit, label=(f'Train' if train else f'Validation'))

        self.plots_from_logit_solution(solution, train=train, block_header=self.first_test_logit[1],
                                       nonce_labels=self.first_test_logit[1])
        return

            
    #
    # ====================================================================
    def augment_data_during_training(self, block_header: torch.Tensor, decoder_labels: torch.Tensor) -> tuple:
        block_header_extra = []
        decoder_labels_extra = []
        delta = (1+self.trainer.current_epoch)
        """
        #  Changes the N_once Values
        for i in range(self.n_block_tail):
            block_header_augment = block_header.clone()
            block_header_augment[:,~i] = (block_header_augment[:,~i] + delta) % (self.vocab_size-1)
            decoder_labels_augment = self.labelr(block_header_augment, from_tokens=True, reply_with_hash_tokens=True).to(device=decoder_labels.device)
            block_header_extra.append(block_header_augment)
            decoder_labels_extra.append(decoder_labels_augment)
        """

        mrkl_start = 4+32-1
        time_end = mrkl_start + 32+4

        block_header_augment = block_header.clone()
        block_header_augment[:, mrkl_start:time_end] = torch.randint_like(block_header_augment[:, mrkl_start:time_end], 0, 255)
        decoder_labels_augment = self.labelr(block_header_augment, from_tokens=True, reply_with_hash_tokens=True).to(device=decoder_labels.device)
        block_header_extra.append(block_header_augment)
        decoder_labels_extra.append(decoder_labels_augment)

        block_header = torch.concat( [block_header, *block_header_extra], dim=0 )
        decoder_labels = torch.concat( [decoder_labels, *decoder_labels_extra], dim=0 )
        #print(f"{block_header.shape}")
        #print(f"{decoder_labels.shape}")
        del block_header_augment, decoder_labels_augment
        del block_header_extra, decoder_labels_extra
        return (block_header, decoder_labels)
    
    # ######################################################################################
    # ######################################################################################
    # ------------------------------      OPTIMIZER      -----------------------------------
    # ======================================================================================
    # ######################################################################################
    def configure_optimizers(self):
        """  configure our Adam optimizers using the learning rates and betas we saved in self.hparams. """
        """ LR = 0.0000075 """

        """ LR = 0.00266 ->  L1 loss stops decreasing and begins increasing. 
        (params are getting bigger faster than L1 can shrink them.)"""
        """
        optimizer = torch.optim.RMSprop(self.parameters(),
                                      lr=1.17e-3,  #0.00004, #0.000375/55,
                                      momentum=0.9,
                                      #betas=(0.5, 0.98),
                                      #eps=1.0e-8,
                                      weight_decay=self.decay_rate)
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.max_learning_rate,  #1.17e-3,  #0.00004, #0.000375/55,
                                      betas=(0.95, 0.999),
                                      #betas=(0.5, 0.98),
                                      eps=1.0e-8,
                                      weight_decay=self.decay_rate)
        
        """ """
        sched = CosineWarmupScheduler(optimizer, warmup=self.warmup_steps, max_iters=(self.trainer.max_epochs*self.n_data_batches) )

        lr_scheduler_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }
        return lr_scheduler_config


#
# ==================================================================#==================================================
# ==================================================================#==================================================
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """  max_iters: (max_epochs * len(training_dataloader))

    """
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)
    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor



##########################
if __name__ == '__main__':
    pass
##########################
