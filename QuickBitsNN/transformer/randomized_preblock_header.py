import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from datetime import timedelta, datetime
import pytorch_lightning as pl
from ..data_io.utils import load_json_file, save_json_file
from ..framework.functions import BitsZeros
from ..framework.labeler import Labeler
from ..framework.tokenizer import Tokenizer


# =====---------------------------====================
class Randomized_Preblock_Header(pl.LightningModule):
    def __init__(self, saved_filename=None, preblock_stats=None):
        super().__init__()
        self.encoder_input_dim = 76
        self.decoder_input_dim = 4
        self.labelr = Labeler()
        self.tokenizr = Tokenizer()

        data = {}
        # -----------------------------
        if saved_filename is not None:
            bits_generated_labels = load_json_file(os.getcwd() + f"/data/" + f"{saved_filename}")
        else:
            bits_generated_labels = load_json_file(
                os.getcwd() + f"/data/" + 'List_of_Bits_Mapped_to_nLeading_Hash0s.json')

        # -----------------------------
        if preblock_stats is not None:
            data = preblock_stats
        else:
            data = load_json_file(
                os.getcwd() + f"/data/" + 'Mean_and_Std_Token_IDs_for_Training_QuickBits_(val_rnd_200_unique).json')

        self.register_buffer("PREBLOCK_TOKEN_AVERAGES", torch.tensor(data["mean_tokens"]))
        self.register_buffer("PREBLOCK_TOKEN_STD", torch.tensor(data["std_tokens"]))
        self.register_buffer("PREBLOCK_TOKEN_UNIQUE_BITS", torch.tensor(data["unique_int_bits"], dtype=torch.long))

        # -----------------------------
        logit_class_pdf = load_json_file(os.getcwd() + f"/data/" + f'Nonce_Token_PDF_by_digit.json')
        self.register_buffer("LOGIT_CLASS_PDF", torch.tensor(logit_class_pdf))

        # -----------------------------
        version_values = load_json_file(os.getcwd() + f"/data/" + f'full_quickbit_unique_versions.json')
        self.allowable_versions = [self.tokenizr.tokenize(self.labelr._int_to_tokenable_hex(int(ver_))) for ver_ in
                                   version_values]

        # -----------------------------
        bits_values = load_json_file(os.getcwd() + f"/data/" + f'full_quickbit_unique_bits.json')
        self.allowable_bits = [self.tokenizr.tokenize(self.labelr._int_to_tokenable_hex(int(bit_))) for bit_ in
                               bits_values]

        self.zeros_to_bits = {}
        _ = [self.zeros_to_bits.setdefault(BitsZeros(bit_), []) for bit_ in self.PREBLOCK_TOKEN_UNIQUE_BITS]
        _ = [self.zeros_to_bits[BitsZeros(bit_)].append(self.tokenizr.tokenize(self.labelr._int_to_tokenable_hex(bit_)))
             for bit_ in self.PREBLOCK_TOKEN_UNIQUE_BITS]

        for b_ in bits_generated_labels.keys():
            self.zeros_to_bits.setdefault(int(b_), [])
            for int_bit_ in bits_generated_labels[b_]:
                self.zeros_to_bits[int(b_)].append(self.tokenizr.tokenize(self.labelr._int_to_tokenable_hex(int_bit_)))

    # -----------------------------------------
    def generate(self, leading_zeros=16, samples=1):
        # print(f"{self.zeros_to_bits[leading_zeros]}")
        zbits = torch.stack(random.choices(self.zeros_to_bits[leading_zeros], k=samples))

        preblocks = [torch.normal(self.PREBLOCK_TOKEN_AVERAGES, self.PREBLOCK_TOKEN_STD) for i in range(samples)]
        preblocks = [torch.remainder(tkns_, 255).to(dtype=torch.long) for tkns_ in preblocks]
        preblocks = torch.stack(preblocks)

        preblocks[:, -8:-4] = zbits
        return preblocks

    # ------------------------------------------------------------------------------------
    def generate_from_class_pdf(self, leading_zeros=16, samples=1):
        next_token = torch.zeros((samples, 80,), dtype=torch.long, device=self.LOGIT_CLASS_PDF.device)
        letters_, classes_ = self.LOGIT_CLASS_PDF.shape

        # Full Header
        for idx_ in range(letters_):
            next_token[:, idx_] = torch.multinomial(self.LOGIT_CLASS_PDF[idx_].repeat(samples, 1),
                                                    num_samples=1).squeeze(-1)  # Sample

        # Version
        version_tokens = random.choices(self.allowable_versions, k=samples)
        next_token[:, :4] = torch.stack(version_tokens, dim=0)

        # Time
        time_tokens = [self.select_random_future_time() for s_ in range(samples)]
        next_token[:, -12:-8] = torch.stack(time_tokens, dim=0)

        # Bits
        if leading_zeros is None:
            zbits_tokens = torch.stack(random.choices(self.allowable_bits, k=samples))
        else:
            zbits_tokens = torch.stack(random.choices(self.zeros_to_bits[leading_zeros], k=samples))
        next_token[:, -8:-4] = zbits_tokens

        del version_tokens, time_tokens, zbits_tokens
        return next_token.to(dtype=torch.long)

    # ------------------------------------------------------------------------------------
    def select_random_future_time(self):
        # Pick a random time between Now and the Future
        unix_atm = int(time.time())
        unix_last_mined     = 5390942400
        unix_one_year       =   31536000
        unix_one_month      =    2628000
        unix_one_week       =     604800
        unix_five_days      =     432000
        unix_three_days     =     259200
        unix_two_days       =     172800
        unix_ten_mins       =        600

        # Generate Random Time between now and future time
        # its_been = (unix_one_week) + unix_five_days + unix_three_days - unix_ten_mins + unix_two_days
        # random_tokens = torch.randint(unix_atm, (unix_atm+(unix_one_month)+unix_ten_mins+1), (1,))
        #random_time = torch.randint(unix_atm, unix_last_mined, (1,))
        random_time = torch.randint(unix_atm, unix_atm+(2*unix_one_year), (1,))

        # Retokenize New Random Time
        random_time_token_ids = self.tokenizr.tokenize(
            self.labelr._int_to_tokenable_hex(int(random_time.item()))
        )

        return random_time_token_ids

    # ------------------------------------------------------------------------------------
    def fake_data(self, leading_zeros=[None], samples=1, from_class_pdf=True, reply_headers=False):
        """ Input: leading_zeros: None, int, or list(ints).
                    samples: int  (Number of fake data items for each leading_zeros)
            Returns:  (preblock_header, preblock_tail)
                tuple( torch.tensor().shape = [samples*len(leading_zeros), src_tokens],
                            torch.tensor().shape = [samples*len(leading_zeros), tgt_tokens] )
        """

        if isinstance(leading_zeros, int):
            b_zeros = [leading_zeros]
        elif isinstance(leading_zeros, list):
            b_zeros = leading_zeros

        if from_class_pdf:
            rnd_preblock = [self.generate_from_class_pdf(leading_zeros=lz_, samples=samples) for lz_ in b_zeros]
        else:
            rnd_preblock = [self.generate(leading_zeros=lz_, samples=samples) for lz_ in b_zeros]

        rnd_preblock = torch.concat(rnd_preblock, dim=0)

        if reply_headers: return rnd_preblock

        rnd_block_header = rnd_preblock[:, :self.encoder_input_dim]
        rnd_block_tail = rnd_preblock[:, -(self.decoder_input_dim):]
        return (rnd_block_header, rnd_block_tail)

    # ------------------------------------------------------------------------------------
    def create_augmented_dataset(self, batch_size=16, max_data=20000, leading_zeros=[4, None]) -> None:
        """ Generates Self Consistent Preblocks by Keeping Only Accepted Randomly Sampled Headers. """

        file_name = os.getcwd() + f"/data/" + f"Fake_Augmented_Headers.json"
        try:
            augmented_dataset = load_json_file(file_name)
            n_init = len(augmented_dataset)
        except:
            augmented_dataset = []
            n_init = 0

        n_aug = 0
        n_loops = 0
        total_samples_checked = 0
        og_start_time = datetime.now()
        start_time = og_start_time
        print(f'Started Generating Samples... \t {start_time}')

        while n_aug < max_data:
            n_aug = len(augmented_dataset)
            fake_headers = self.fake_data(leading_zeros=leading_zeros, samples=batch_size, reply_headers=True)
            accepted_ids = self.labelr(fake_headers, from_tokens=True, reply_with_acceptance=True)
            accepted_headers = fake_headers[accepted_ids]
            total_samples_checked += batch_size * len(leading_zeros)
            n_loops += 1

            if len(accepted_headers) > 0:
                detokenized_headers = [self.tokenizr.detokenize(header_) for header_ in accepted_headers.unbind()]
                augmented_dataset.extend(detokenized_headers)

                n_items = len(augmented_dataset)
                time_for_all_solve = (datetime.now() - og_start_time).seconds
                time_to_solve = (datetime.now() - start_time).seconds
                start_time = datetime.now()
                est_hash_zeros = np.emath.logn(16, total_samples_checked)
                print(
                    f'Headers Solved: {n_items:>6}/{total_samples_checked:>7} \t Time: {time_to_solve}(s) \t Item/Sec: {time_for_all_solve / (n_items - n_init):.3f} \t Log16: {est_hash_zeros:.3f}(z) \t nLoops: {n_loops}')
                save_json_file(augmented_dataset, file_name)
                n_loops = 0

            if (n_loops % (16 ** 4)) == 0:
                print(f'Headers Checked: {total_samples_checked}')
                # n_loops = 0
        return
