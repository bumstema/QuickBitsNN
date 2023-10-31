################################################################
################################################################
import numpy as np
import torch

import  json
import  base64
import  hashlib
import  binascii

from typing             import List, Dict, Any
from binascii           import unhexlify, hexlify
from dataclasses        import dataclass, asdict, field
from torch.utils.data   import DataLoader, Dataset

from ..data_io.utils        import get_device
from ..framework.constants  import BATCH_SAMPLES
from ..framework.constants  import START_TOKEN, END_TOKEN, PADDING_TOKEN, INDEX_TO_HASHEX, HASHEX_VOCABULARY



#  Define Each Sub-Component of the BlockChain Block
#--------------------------------------------------------------
#--------------------------------------------------------------
@dataclass
class Input:
    sequence: int
    witness: str
    script: str
    index: int
    prev_out: dict

#--------------------------------------------------------------
@dataclass
class Output:
    type: int
    spent: bool
    value: int
    spending_outpoints: List[dict]
    n: int
    tx_index: int
    script: str
    addr: str

#--------------------------------------------------------------
@dataclass
class Transaction:
    hash: str
    ver: int
    vin_sz: int
    vout_sz: int
    size: int
    weight: int
    fee: int
    relayed_by: str
    lock_time: int
    tx_index: int
    double_spend: bool
    time: int
    block_index: int
    block_height: int
    inputs: List[Input]
    out: List[Output]


#  Main Class to Store Parameters and Calculate Hashes
#--------------------------------------------------------------
#--------------------------------------------------------------
@dataclass
class QuickBit:
    hash: str
    ver: int
    prev_block: str
    mrkl_root: str
    time: int
    bits: int
    next_block: List[str]
    fee: int
    nonce: int
    n_tx: int
    size: int
    block_index: int
    main_chain: bool
    height: int
    weight: int
    tx: List[Transaction]
    
    mediantime : int
    difficulty : int
    chainwork : str
    versionHex : str
    confirmations : int
    nTx : int
    
     #  ----- ⭐️ -----
    def nonce_hex_string(self):
        return self.little_endian( self.four_byte_hex(self.nonce)).decode('utf-8')
        
    #  ----- ⭐️ -----
    def doubleSha256(self, hex):
       bin = binascii.unhexlify(hex)
       hash = hashlib.sha256(bin).digest()
       hash2 = hashlib.sha256(hash).digest()
       return binascii.hexlify(hash2)

    #  ----- ⭐️ -----  # Converting to little-endian hex notation
    def little_endian(self, value ):
        return binascii.hexlify(binascii.unhexlify( value )[::-1])

    #  ----- ⭐️ -----
    def big_endian(self, hash):
        return str( binascii.hexlify(binascii.unhexlify( hash )[::-1]) ,"ascii")

    #  ----- ⭐️ -----  # 4-byte hex notation
    def four_byte_hex(self, number ):
        return hex(int(0x100000000)+ number )[-8:]

    #  ----- ⭐️ -----
    def build_header_bits(self, decode=True):
        hex_bits        = self.little_endian( self.four_byte_hex(self.bits) )
        if decode:
            return (hex_bits).decode('utf-8')
        return hex_bits

    #  ----- ⭐️ -----
    def build_header_time(self, decode=True):
        hex_time        = self.little_endian( self.four_byte_hex(self.time) )
        if decode:
            return (hex_time).decode('utf-8')
        return hex_time

    #  ----- ⭐️ -----
    def build_header_ver(self, decode=True):
        hex_ver         = self.little_endian( self.four_byte_hex(self.ver) )
        if decode:
            return (hex_ver).decode('utf-8')
        return hex_ver
        
    #  ----- ⭐️ -----
    def build_header_merkleroot(self, decode=True):
        hashMerkleRoot  = self.little_endian(self.mrkl_root)
        if decode:
            return (hashMerkleRoot).decode('utf-8')
        return hashMerkleRoot
        
    #  ----- ⭐️ -----
    def build_header_nonce(self, decode=True):
        hex_nonce  = self.little_endian(self.four_byte_hex(self.nonce))
        if decode:
            return (hex_nonce).decode('utf-8')
        return hex_nonce


    #  ----- ⭐️ -----
    def build_nonvariable_header(self, decode=True):
        hex_ver         = self.little_endian( self.four_byte_hex(self.ver) )
        hashPrevBlock   = self.little_endian(self.prev_block)
        hashMerkleRoot  = self.little_endian(self.mrkl_root)
        if decode:
            return (hex_ver+hashPrevBlock+hashMerkleRoot).decode('utf-8')
        return hex_ver+hashPrevBlock+hashMerkleRoot

    #  ----- ⭐️ -----
    def build_nonceless_header(self, decode=False):
        hex_ver         = self.little_endian( self.four_byte_hex(self.ver) )
        hashPrevBlock   = self.little_endian(self.prev_block)
        hashMerkleRoot  = self.little_endian(self.mrkl_root)
        hex_time        = self.little_endian( self.four_byte_hex(self.time) )
        hex_bits        = self.little_endian( self.four_byte_hex(self.bits) )
        if decode:
            return (hex_ver+hashPrevBlock+hashMerkleRoot+hex_time+hex_bits).decode('utf-8')
        return hex_ver+hashPrevBlock+hashMerkleRoot+hex_time+hex_bits

    #  ----- ⭐️ -----
    def build_header(self, decode=False):
        # Turn into proper (4/32/32/4/4/4) hex notation & convert into little-endian hex notation
        hex_ver         = self.little_endian( self.four_byte_hex(self.ver) )
        hashPrevBlock   = self.little_endian(self.prev_block)
        hashMerkleRoot  = self.little_endian(self.mrkl_root)
        hex_time        = self.little_endian( self.four_byte_hex(self.time) )
        hex_bits        = self.little_endian( self.four_byte_hex(self.bits) )
        hex_nonce       = self.little_endian( self.four_byte_hex(self.nonce))
        if decode:
            return (hex_ver+hashPrevBlock+hashMerkleRoot+hex_time+hex_bits+hex_nonce).decode('utf-8')
        return hex_ver+hashPrevBlock+hashMerkleRoot+hex_time+hex_bits+hex_nonce
        
    #  ----- ⭐️ -----
    def calculate_header_mining_hash(self,  **kwargs):
        header=None
        if header == None : header = self.build_header(nonce=self.nonce, time=self.time)
        header  = binascii.unhexlify(header)
        hash    = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        hash    = binascii.hexlify(hash)
        hash    = binascii.hexlify(binascii.unhexlify(hash)[::-1])
        return hash.decode('utf-8')

    #  ----- ⭐️ -----
    def calculate_mining_hash(self, nonce, header = None):
        if header == None : header = self.build_nonceless_header()
        hex_nonce       = self.little_endian( self.four_byte_hex(nonce) )
        header  += hex_nonce
        header  = binascii.unhexlify(header)
        hash    = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        hash    = binascii.hexlify(hash)
        hash    = binascii.hexlify(binascii.unhexlify(hash)[::-1])
        return hash.decode('utf-8')


    #  ----- ⭐️ -----
    def leading_zeros(self, input_string : str ) -> int :
        zeros = lambda s: len(s) - len(s.lstrip('0'))
        return zeros( input_string )

    #  ----- ⭐️ -----
    def hash_zeros(self ) -> int :
        zeros = lambda s: len(s) - len(s.lstrip('0'))
        return zeros( self.hash )

    #  ----- ⭐️ -----
    def leading_zeros_of_hash(self, nonce):
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        hash = self.calculate_mining_hash( nonce)
        hash_zeros = self.leading_zeros(hash)
        torch.tensor(hash_zeros, device = device)
        return hash_zeros

    #  ----- ⭐️ -----
    def leading_zeros_of_hash(self, **kwargs):
        for variable_name, default_value in self.__dataclass_fields__.items():
            setattr(self, variable_name, kwargs.get(variable_name, getattr(self, variable_name)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hash = self.calculate_header_mining_hash( nonce=self.nonce, time=self.time)
        hash_zeros = self.leading_zeros(hash)
        return hash_zeros

    #  ----- ⭐️ -----
    def calculate_merkle_root(self):
        if len(self.tx) == 0:
            return None

        transaction_hashes = [ self.little_endian(transaction['hash']) for transaction in self.tx ]
        while len(transaction_hashes) > 1:
            if len(transaction_hashes) % 2 != 0:
                transaction_hashes.append(transaction_hashes[-1])
            next_level = []
            for i in range(0, len(transaction_hashes), 2):
                combined_hash = transaction_hashes[i] + transaction_hashes[i + 1]
                next_level.append(self.doubleSha256( combined_hash ))
            transaction_hashes = next_level

        return  self.big_endian(transaction_hashes[0])

    #  ----- ⭐️ -----
    def calculate_transaction_hashes(self):
        transactions = self.tx
        serialized_transaction = json.dumps(transactions)
        return hashlib.sha256(serialized_transaction.encode()).hexdigest()

    #  ----- ⭐️ -----
    def reverse_four_byte_hex(self, hex_number ):
        return int(hex(f'0x1{hex_number}'), 16)

    def parameters(self):
        return {'ver':self.ver, 'prev_block':self.prev_block, 'mrkl_root':self.mrkl_root, 'time':self.time, 'bits':self.bits, 'nonce':self.nonce}



#----------------------------------------------------------------
def save_quickbits(data, filename):
    serialized_data = [quickbit.__dict__ for quickbit in data]
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump(serialized_data, f, indent=4, default=lambda o: o.__dict__)

#----------------------------------------------------------------
def load_quickbits(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
        data = [ preprocess_quickbit_json( item ) for item in data]
        return [QuickBit(**quickbit) for quickbit in data]


#--------------------------------------------------------------
class QuickBitsTransformerDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.x_data = input_data
        self.y_data = target_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

#--------------------------------------------------------------
@dataclass
class QuickBitsModelDataLoader:
    train       : DataLoader
    validate    : DataLoader
    test        : DataLoader


#--------------------------------------------------------------
@dataclass
class QuickBitsDataCart:
    train       : List = field(default_factory = lambda: ([]))
    validate    : List = field(default_factory = lambda: ([]))
    test        : List = field(default_factory = lambda: ([]))


#----------------------------------------------------------------
def preprocess_quickbit_json(json_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Remove unwanted fields from the transaction objects
    
    
    #  From RCP json format
    json_dict.setdefault('mediantime',None)
    json_dict.setdefault('difficulty',None)
    json_dict.setdefault('chainwork',None)
    json_dict.setdefault('versionHex',None)
    json_dict.setdefault('confirmations',None)
    json_dict.setdefault('nTx',None)
    json_dict['tx'] = []
    
    
    # Discard unneeded info for Hash Calculation
    """
    for transaction in json_dict['tx']:
        transaction.pop('ver', None)
        transaction.pop('vin_sz', None)
        transaction.pop('vout_sz', None)
        transaction.pop('size', None)
        transaction.pop('weight', None)
        transaction.pop('fee', None)
        transaction.pop('relayed_by', None)
        transaction.pop('lock_time', None)
        transaction.pop('tx_index', None)
        transaction.pop('double_spend', None)
        transaction.pop('time', None)
        transaction.pop('block_index', None)
        transaction.pop('block_height', None)
        transaction.pop('inputs', None)
        for output in transaction['out']:
            output.pop('type', None)
            #output.pop('spent', None)
            output.pop('spending_outpoints', None)
            output.pop('n', None)
            output.pop('tx_index', None)
            output.pop('script', None)
            output.pop('addr', None)
            output.pop('value', None)
    """
    return json_dict


    

#  Deeper Explanation of Each Classes' Variables
#--------------------------------------------------------------
#--------------------------------------------------------------

"""
    hash: The hash of the block.
    ver: The block version.
    prev_block: The hash of the previous block in the blockchain.
    mrkl_root: The Merkle root of the transactions in the block.
    time: The timestamp of the block creation, represented in Unix time format.
    bits: The difficulty target for the block.
    next_block: The hash of the next block in the blockchain (if available).
    fee: The total transaction fee collected by the miner for including the transactions in the block.
    nonce: The nonce value used in the block's proof-of-work calculation.
    n_tx: The number of transactions in the block.
    size: The size of the block in bytes.
    block_index: The index of the block.
    main_chain: Indicates whether the block is part of the main blockchain (True) or an orphaned block (False).
    height: The height of the block in the blockchain.
    weight: The weight of the block.
    tx: A list of transactions included in the block.

Within the tx list, each item represents a transaction. Here's the explanation of the transaction-related entries:

    hash: The hash of the transaction.
    ver: The transaction version.
    vin_sz: The number of inputs in the transaction.
    vout_sz: The number of outputs in the transaction.
    size: The size of the transaction in bytes.
    weight: The weight of the transaction.
    fee: The transaction fee.
    relayed_by: The IP address of the node that relayed the transaction.
    lock_time: The transaction's lock time.
    tx_index: The index of the transaction.
    double_spend: Indicates whether the transaction has been double spent (True) or not (False).
    time: The timestamp of the transaction, represented in Unix time format.
    block_index: The index of the block containing the transaction.
    block_height: The height of the block containing the transaction.
    inputs: A list of transaction inputs.
    out: A list of transaction outputs.

Within the inputs list, each item represents an input to the transaction. Here's the explanation of the input-related entries:

    sequence: The sequence number of the input.
    witness: The witness data for the input.
    script: The input script.
    index: The index of the input.
    prev_out: Information about the previous transaction output spent by this input.

Within the out list, each item represents an output of the transaction. Here's the explanation of the output-related entries:

    type: The output type.
    spent: Indicates whether the output has been spent (True) or not (False).
    value: The value of the output.
    spending_outpoints: Information about the outputs spending this output.
    n: The index of the output.
    tx_index: The index of the transaction containing the output.
    script: The output script.
    addr: The Bitcoin address associated with the output.
"""

'''.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.'''
"""
To compute the hash of a Bitcoin block, several components are used. Here are the main parts used in the hash calculation:

    ver (Version): The version number of the block.
    prev_block (Previous Block Hash): The hash of the previous block in the blockchain.
    mrkl_root (Merkle Root): The Merkle root hash of all the transactions in the block.
    time (Timestamp): The timestamp of the block creation.
    bits (Difficulty Target): The difficulty target for the block.
    nonce: The nonce value, which miners adjust during the mining process to find a hash that meets the difficulty requirement.
"""
"""
@dataclass
class BlockHeader():


def BlockHeaderFromGet(json_dict):
    {                                          (json object)
  json_dict["version"] : n,                           (numeric) The preferred block version
  json_dict["rules"] : ["str"],
  json_dict["vbavailable"] : {                        (json object) set of pending, supported versionbit (BIP 9) softfork deployments
    "rulename" : n,                        (numeric) identifies the bit number as indicating acceptance and readiness for the named softfork rule
    ...
  },
  json_dict["vbrequired"] : n,                        (numeric) bit mask of versionbits the server requires set in submissions
  json_dict["previousblockhash"] : "str",             (string) The hash of current highest block
  json_dict["transactions"] : [                       (json array) contents of non-coinbase transactions that should be included in the next block
    {                                      (json object)
      "data" : "hex",                      (string) transaction data encoded in hexadecimal (byte-for-byte)
      "txid" : "hex",                      (string) transaction id encoded in little-endian hexadecimal
      "hash" : "hex",                      (string) hash encoded in little-endian hexadecimal (including witness data)
      "depends" : [                        (json array) array of numbers
        n,                                 (numeric) transactions before this one (by 1-based index in 'transactions' list) that must be present in the final block if this one is
        ...
      ],
      "fee" : n,                           (numeric) difference in value between transaction inputs and outputs (in satoshis); for coinbase transactions, this is a negative Number of the total collected block fees (ie, not including the block subsidy); if key is not present, fee is unknown and clients MUST NOT assume there isn't one
      "sigops" : n,                        (numeric) total SigOps cost, as counted for purposes of block limits; if key is not present, sigop cost is unknown and clients MUST NOT assume it is zero
      "weight" : n                         (numeric) total transaction weight, as counted for purposes of block limits
    },
    ...
  ],
  json_dict["coinbaseaux"] : { "key" : "hex" },                      (json object) data that should be included in the coinbase's scriptSig content (string) values must be in the coinbase (keys may be ignored)
 
  json_dict["coinbasevalue"] : n,                     (numeric) maximum allowable input to coinbase transaction, including the generation award and transaction fees (in satoshis)
  json_dict["longpollid"] : "str",                    (string) an id to include with a request to longpoll on an update to this template
  json_dict["target"] : "str",                        (string) The hash target
  json_dict["mintime"] : xxx,                         (numeric) The minimum timestamp appropriate for the next block time, expressed in UNIX epoch time
  json_dict["mutable"] : [  "str" ],                          (json array) list of ways the block template may be changed (string) A way the block template may be changed, e.g. 'time', 'transactions', 'prevblock'
  json_dict["noncerange"] : "hex",                    (string) A range of valid nonces
  json_dict["sigoplimit"] : n,                        (numeric) limit of sigops in blocks
  json_dict["sizelimit"] : n,                         (numeric) limit of block size
  json_dict["weightlimit"] : n,                       (numeric) limit of block weight
  json_dict["curtime"] : xxx,                         (numeric) current timestamp in UNIX epoch time
  json_dict["bits"] : "str",                          (string) compressed target of next block
  json_dict["height"] : n,                            (numeric) The height of the next block
  json_dict["default_witness_commitment"] : "str"     (string, optional) a valid witness commitment for the unmodified block template
}
"""

