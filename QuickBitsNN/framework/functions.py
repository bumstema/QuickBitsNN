################################################################
import json
import hashlib
import binascii
import numpy as np
import torch

from typing import List, Dict, Any


#----------------------------------------------------------------
#----------------------------------------------------------------
def little_endian( value ):
    return binascii.hexlify(binascii.unhexlify( value )[::-1])
  
#----------------------------------------------------------------
def big_endian( hash):
    return str( binascii.hexlify(binascii.unhexlify( hash )[::-1]) ,"ascii")

#----------------------------------------------------------------
def four_byte_hex( number ):
    return hex(int(0x100000000)+ number )[-8:]

#----------------------------------------------------------------
def little_endian_hex_to_int(hex_string):
    """ Decodes the integer value from the preblock hex representation. """
    # Reverse the little-endian hex string by bytes
    reversed_hex = ''.join(reversed([hex_string[i:i+2] for i in range(0, len(hex_string), 2)]))
    
    # Convert the reversed hex string to base 10 integer
    base_10_integer = int(reversed_hex, 16)
    return base_10_integer

#----------------------------------------------------------------
def HeaderNonceMiningHash(prenonce_header, nonce):

    header  = prenonce_header + nonce
    if len(header) != 160: print(f"{prenonce_header = }  {nonce = } {len(header) = }")
    header  = binascii.unhexlify(header)
    hash    = hashlib.sha256(hashlib.sha256(header).digest()).digest()
    hash    = binascii.hexlify(hash)
    hash    = binascii.hexlify(binascii.unhexlify(hash)[::-1])
    return hash.decode('utf-8')

#----------------------------------------------------------------
def construct_block_header(version, prev_block_hash, merkle_root, timestamp, difficulty_target, nonce):
    block_header = (
        version.to_bytes(4, byteorder='little') +
        bytes.fromhex(prev_block_hash) +
        bytes.fromhex(merkle_root) +
        int(timestamp).to_bytes(4, byteorder='little') +
        int(difficulty_target).to_bytes(4, byteorder='little') +
        int(nonce).to_bytes(4, byteorder='little')
    )
    return block_header

#----------------------------------------------------------------
def LeadingZeros( input_string : str ) -> int :
    zeros = lambda s: len(s) - len(s.lstrip('0'))
    return zeros(input_string)

#----------------------------------------------------------------
def BitsZeros(bits):
    """ Decodes the required leading zeros from the bits parameter. """
    #hex_bits = int(int(bits,10),16)
    hex_bits = four_byte_hex(int(bits))
    bits_exponent = str(hex_bits)[:2]
    bits_exponent = int(f'0x'+bits_exponent,16)
    required_zeros = 2 * (32 - bits_exponent)
    return required_zeros

#----------------------------------------------------------------
def HexValueAsPercent( hexval, maxbit ):
    return (float( int( str(hexval), 16 ) ) / ((16 ** maxbit) - 1))

#----------------------------------------------------------------
def PreblockPercents( bit ):
    data = [(bit.ver, 8), (bit.prev_block, 64), (bit.mrkl_root, 64), (bit.time, 8), (bit.bits, 8), (bit.nonce, 8)]
    percents =  [HexValueAsPercent( *i ) for i in data]
    return percents

#----------------------------------------------------------------
def LetterHexValuesFromWord(word):
    hex_letters = []
    for letter in word:
        hex_letters.extend( [(int(letter,16)+1) / 16] )
    return np.array(hex_letters)


#----------------------------------------------------------------
def TwoStringConvolutionMatrix(str1, str2):
    matrix_stack = []
    main_str = "0123456789abcdef"
    l_neighbours = "123456789abcdef0"
    r_neighbours = "f0123456789abcde"
    characters = list(map(list, zip(*(main_str, l_neighbours, r_neighbours))))
    
    for hex_val, R, L in characters:
        matrix = np.zeros((len(str2), len(str1)), dtype=np.float32)

        hex_val_int = 1 #int(hex_val, 16)
        R_val = - hex_val_int / 2
        L_val =  hex_val_int / 2
        
        matrix[(str2 == hex_val) & (str1 == hex_val)] = hex_val_int
        matrix[(str2 == R) & (str1 == R)] = R_val
        matrix[(str2 == L) & (str1 == L)] = L_val
        
        matrix_stack.append(matrix)
    
    return np.array(matrix_stack)



#----------------------------------------------------------------
#----------------------------------------------------------------
