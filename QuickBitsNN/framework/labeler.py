import torch
import torch.nn as nn
import hashlib
import binascii
from typing import List, Dict
import copy
from ..framework.tokenizer import Tokenizer
#from ..framework.tokenizer_single_digit import Tokenizer


class Labeler(nn.Module):
    """ Takes in a batched set of sequences and calculates the hash for each. """
    """ Returns a list of True/False for in the sequences satisfy blockchain acceptance. """
    def __init__(
        self,
    ):
        super().__init__()
        self.tokenizer = Tokenizer()

    def forward(
        self,
        x,
        from_hash=False,
        from_preblock=False,
        from_tokens=False,
        from_gray=False,
        reply_with_acceptance=False,
        reply_as_target_ratio=False,
        reply_with_hash_tokens=False
    ):
        if from_hash:
            return torch.tensor([self._leading_zeros(x_) for x_ in x], dtype=torch.long)

        if from_preblock:
            x_preblock = copy.deepcopy(x)
            x = [self._mining_hash(x_) for x_ in x]
            if reply_with_acceptance:
                return self._acceptance_from_target(x_preblock, x)
            return torch.tensor([self._leading_zeros(x_) for x_ in x], dtype=torch.long)

        if from_tokens:
            x = [self._detokenize(x_) for x_ in x]
            x_preblock = copy.deepcopy(x)
            x = [self._mining_hash(x_) for x_ in x]
            if reply_with_hash_tokens:
                return torch.stack([self.tokenizer.tokenize(x_) for x_ in x], dim=0)
            if reply_as_target_ratio:
                return self._acceptance_target_ratio(x_preblock, x)
            if reply_with_acceptance:
                return self._acceptance_from_target(x_preblock, x)
            return torch.tensor([self._leading_zeros(x_) for x_ in x], dtype=torch.long)


        with torch.no_grad():
            x = x.detach()
            x = [self._detokenize(x_) for x_ in x]
            x_preblock = copy.deepcopy(x)
            x = [self._mining_hash(x_) for x_ in x]
            if reply_with_acceptance:
                return self._acceptance_from_target(x_preblock, x)
            return torch.tensor([self._leading_zeros(x_) for x_ in x], dtype=torch.long)
    
    
    # ==============================================================
    
    
    def _detokenize(self, tokens: torch.Tensor) -> str:
        header = self.tokenizer.detokenize(tokens)
        return header


    def _mining_hash(self, header: str) -> str:
        # header  = prenonce_header + nonce
        if len(header) != 160:
            print(f"{header[:152] = }  {header[152:] = } {len(header) = }")
        header = binascii.unhexlify(header)
        hash = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        hash = binascii.hexlify(hash)
        hash = binascii.hexlify(binascii.unhexlify(hash)[::-1])
        return hash.decode("utf-8")


    def _leading_zeros(self, hash: str) -> int:
        zeros = lambda s: len(s) - len(s.lstrip("0"))
        return zeros(hash)


    def _bits_zeros(self, hex_str: str) -> int:
        """Decodes the required leading zeros from the Bits parameter."""
        """  Assumes the input has already been detokenized.          """
        # hex_bits = int(int(bits,10),16)

        reversed_hex = "".join(
            reversed([hex_str[i : i + 2] for i in range(0, len(hex_str), 2)])
        )
        # Convert the reversed hex string to base 10 integer
        raw_bits = int(reversed_hex, 16)
        hex_bits = hex(int(0x100000000) + (int(raw_bits)))[-8:]
        bits_exponent = str(hex_bits)[:2]
        bits_exponent = int(f"0x" + bits_exponent, 16)
        required_zeros = 2 * (32 - bits_exponent)
        return required_zeros


    def _little_endian_hex_to_int(self, hex_string: str) -> int:
        """Decodes the integer value from the preblock hex representation."""
        # Reverse the little-endian hex string by bytes
        reversed_hex = "".join(
            reversed([hex_string[i : i + 2] for i in range(0, len(hex_string), 2)])
        )

        # Convert the reversed hex string to base 10 integer
        base_10_integer = int(reversed_hex, 16)
        return base_10_integer


    def _int_to_tokenable_hex(self, number: int) -> str:
        """Encodes the integer value to the hex representation."""
        val_ = hex(int(0x100000000) + int(number) )[-8:]
        val_ = binascii.hexlify(binascii.unhexlify(val_)[::-1])
        return val_.decode("utf-8")


    @staticmethod
    def _calc_target(bits: str) -> bytes:
        """
        Decompress the target from a compact format.
        """
        bits = bytes.fromhex(bits)

        # Extract the parts.
        byte_length = bits[0] - 3
        significand = bits[1:]

        # Scale the significand by byte_length.
        target = significand + b"\x00" * byte_length

        # Fill in the leading zeros.
        target = b"\x00" * (32 - len(target)) + target

        return target
        

    @staticmethod
    def reverse_hex_byte_order(hex_string: str) -> str:
        # Reverse the little-endian hex string by bytes
        reversed_hex = "".join(
            reversed([hex_string[i: i + 2] for i in range(0, len(hex_string), 2)])
        )
        return reversed_hex


    #@staticmethod
    def _test_hash_to_target_ratio(self, bits_hex: str, hash_hex: str) -> bool:
        """ Compare if bytes hash is less than bytes target. Equals 1 when same, accepted > 1 """
        target = self._calc_target(bits_hex)
        hash = bytes.fromhex(hash_hex)
        ratio = (int.from_bytes(target, 'big') / int.from_bytes(hash, 'big'))

        return ratio


    def _acceptance_target_ratio(self, detokenized_preblock:List[str], hash:List[str]) -> torch.Tensor:
        #bits_hex = [x_[-16:-8] for x_ in detokenized_preblock]
        bits_hex = [self.reverse_hex_byte_order(x_[-16:-8]) for x_ in detokenized_preblock]
        acceptance_ratio = [self._test_hash_to_target_ratio(b_, h_) for b_, h_ in zip(bits_hex, hash)]

        return torch.tensor(acceptance_ratio, dtype=torch.float32).reshape(-1)


    #@staticmethod
    def _test_hash_against_target(self, bits_hex: str, hash_hex: str) -> bool:
        """ Compare if bytes hash is less than bytes target. """
        target = self._calc_target(bits_hex)
        hash = bytes.fromhex(hash_hex)

        return (hash < target)
        

    def _acceptance_from_target(self, detokenized_preblock:List[str], hash:List[str]) -> torch.Tensor:
        #bits_hex = [x_[-16:-8] for x_ in detokenized_preblock]
        bits_hex = [self.reverse_hex_byte_order(x_[-16:-8]) for x_ in detokenized_preblock]
        acceptance = [self._test_hash_against_target(b_, h_) for b_, h_ in zip(bits_hex, hash)]

        return torch.tensor(acceptance, dtype=torch.bool).reshape(-1)

    # ==============================================================