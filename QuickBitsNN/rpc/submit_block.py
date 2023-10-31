import requests
import json
import bitcoinrpc
from bitcoinrpc.authproxy import AuthServiceProxy
from bitcoinlib.services.bitcoind import BitcoindClient
from bitcoinlib.wallets import Wallet
from bitcoinlib.services.services import Service
from bitcoinlib.blocks import Block
from bitcoinlib.transactions import Transaction

from ..framework.classes import  BlockTemplate, QuickBit
from ..framework.functions import save_json_file, load_json_file, get_device
from ..framework.constants import INCOMPLETE_DATA_FILE_PATH, WALLET_FILE_PATH
from ..model.train import LoadSavedModel, LoadSavedCheckpoint




rpc_user = 'example_user'
rpc_password = 'example_password'
rpc_port = 8332
rpc_connection = AuthServiceProxy(f"http://{rpc_user}:{rpc_password}@localhost:{rpc_port}")



#----------------------------------------------------------------
#----------------------------------------------------------------

def GetQuickBitFromBlock():

	d = {"rules": ["segwit"]}

	qb_client = BitcoindClient(base_url=f"http://{rpc_user}:{rpc_password}@localhost:{rpc_port}")

	qb_wallet = Wallet(f'example_wallet')
	print(f" {qb_wallet.keys() = } ")

	# New block to be mined
	new_block_template = qb_client.proxy.getblocktemplate(d)
	print(new_block_template.keys())

	coinbase_transaction = Transaction.create_coinbase(height = 1, value = 6.25)
	coinbase_transaction.inputs[0].scriptSig
	exit()


	# Info of last mined block
	latest_block_hash = qb_client.proxy.getbestblockhash()
	latest_block = qb_client.proxy.getblock(latest_block_hash)
	print(latest_block.keys())
	transactions = latest_block['tx'][:1]


	# --- --- verify transactions
	srv = Service(network="bitcoin")
	count = 0
	count_segwit = 0
	for txid in transactions[:10]:
		count += 1
		t = srv.gettransaction(txid)
		t.verify()
		t.info()
		if t.witness_type != "legacy":
			count_segwit += 1
		if not t.verified:
			print("transaction not verified.")

	print(f" Total Tx: {len(transactions)}, Tx: {count} (Segwit: {count_segwit}) ")

	exit()



#----------------------------------------------------------------
def AnalyzeQuickBitWithModel(qb, model_type):

    checkpoint = LoadSavedCheckpoint(model_type)
    model = LoadSavedModel(checkpoint.checkpoint_file(), model)
    dataloader = LoadDataLoaderWithData([qb], checkpoint)
    
    model.to(get_device()).eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            pred = model(inputs)

    return pred


#----------------------------------------------------------------
def block_template():

	d = {"rules": ["segwit"]}
	block_template = rpc_connection.getblocktemplate(d)
	print(f"\n{block_template.keys() = }")
	print(f"\n{block_template['version'] = }")
	print(f"\n{block_template['rules'] = }")
	print(f"\n{block_template['vbavailable'] = }")
	print(f"\n{block_template['vbrequired'] = }")
	print(f"\n{block_template['target'] = }")
	print(f"\n{block_template['previousblockhash'] = }")
	print(f"\n{block_template['curtime'] = }")
	print(f"\n{block_template['mintime'] = }")
	print(f"\n{block_template['bits'] = }")
	print(f"\n{block_template['noncerange'] = }")
	print(f"\n{block_template['mutable'] = }")

	print(f"\n{block_template['coinbaseaux'] = }")



	coinbase_tx = block_template['coinbasevalue']
	block_data = block_template['mutable'][2]


	print(f"{coinbase_tx = }")
	print(f"{block_data = }")
	# Add signature


	exit()
	submission_response  = rpc_connection.submitblock(block_data)

	if submission_response:
		print(f"Block Successfully Submitted!")
	else:
		print(f"Failed to Submit.")



#----------------------------------------------------------------
def create_coinbase_transaction(coinbase_reward_address, coinbase_value, extra_nonce):
    coinbase_script = (
        bytes.fromhex("03") +  # Length of the following script bytes
        bytes.fromhex(extra_nonce) +
        bytes.fromhex("ffffffffffffffff") +  # Sequence number
        bytes.fromhex("43") +  # Length of the following script bytes
        bytes.fromhex("4104e4e5591ec6e694f73c6da5508d4c620e7f239596cb86b1e6abf81b59c534d376036d7c2dd1d619b8dd760c19822ab7f0db5d27f3c08d5670c58c394e6d745dac")
    )
    coinbase_transaction = (
        int(coinbase_value).to_bytes(8, byteorder='little') +
        bytes.fromhex("01") +  # Number of inputs
        bytes.fromhex("0000000000000000000000000000000000000000000000000000000000000000ffffffff") +
        bytes.fromhex("{:02x}".format(len(coinbase_script))) +
        coinbase_script +
        bytes.fromhex("ffffffff") +  # Number of outputs
        bytes.fromhex("00")  # Locktime
    )
    return coinbase_transaction
    
    
#----------------------------------------------------------------
def add_transaction_to_block(block, transaction):
    block.append(transaction)

#----------------------------------------------------------------
def sign_block_header(block_header, private_key):
    block_header_hash = hashlib.sha256(hashlib.sha256(block_header).digest()).digest()
    # Perform signing using the private key
    # This step involves using the private key to create a digital signature for the block header
    # The signing process is beyond the scope of this example as it requires cryptographic libraries and knowledge


    
#--------------------------------------------------------------
# data structure returned from bitcoind core
@dataclass
class BlockTemplate:
    version: int
    rules: List[str]
    vbavailable: Dict[str, int]
    vbrequired: int
    previousblockhash: str
    transactions: List[Dict[str, object]]
    coinbaseaux: Dict[str, str]
    coinbasevalue: int
    longpollid: str
    target: str
    mintime: int
    mutable: List[str]
    noncerange: str
    sigoplimit: int
    sizelimit: int
    weightlimit: int
    curtime: int
    bits: str
    height: int
    default_witness_commitment: str
    
    @classmethod
    def from_json(cls, json_dict):
        def decode_hex_value(value):
            if isinstance(value, str) and value.startswith("0x"):
                return int(value, 16)
            return value

        decoded_dict = {key: decode_hex_value(value) for key, value in json_dict.items()}
        return cls(**decoded_dict)
        
        

#----------------------------------------------------------------
#----------------------------------------------------------------
