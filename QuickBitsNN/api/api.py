
from flask import Flask, request, jsonify
import sys, os, os.path
import uuid
import json
import datetime
from dataclasses import asdict
from pathlib import Path


from QuickBitsNN.model.train import DeployModel, LoadSavedCheckpoint, LoadSavedModel
from QuickBitsNN.model.torch_transformer import Torch_Transformer
from QuickBitsNN.framework.quickbit import QuickBit
from QuickBitsNN.framework.checkpoint import CheckPoint
from QuickBitsNN.data_io.utils import get_device

app = Flask(__name__)




# File path to store API key data
API_KEYS_FILE = 'api_keys.json'
# File path to store API key data
API_USAGE_FILE = 'api_usage.json'



# Dictionary to store API keys and associated metadata
api_keys = {}
api_usage = {}


#----------------------------------------------------------------
#----------------------------------------------------------------

#----------------------------------------------------------------
# Load API key data from the file
def load_api_keys():
    try:
        with open(API_KEYS_FILE, 'r') as file:
            api_keys.update(json.load(file))
    except FileNotFoundError:
        # File doesn't exist, initialize with an empty dictionary
        api_keys.clear()

#----------------------------------------------------------------
# Load API key data from the file
def load_api_usage():
    try:
        with open(API_USAGE_FILE, 'r') as file:
            api_usage.update(json.load(file))
    except FileNotFoundError:
        # File doesn't exist, initialize with an empty dictionary
        api_usage.clear()
        

#----------------------------------------------------------------
def load_model(checkpoint_info):

    checkpoint_info = LoadSavedCheckpoint(checkpoint_info.checkpoint_file())
    print(asdict(checkpoint_info))
    model = Torch_Transformer(**(asdict(checkpoint_info)['config_mdl']))
    model = LoadSavedModel(checkpoint_info.checkpoint_file(), model)
    return model
    
#----------------------------------------------------------------
def deploy_model(model, input_sequence):
    prediction = DeployModel(model, input_sequence)
    return prediction

#----------------------------------------------------------------
# Save API key data to the file
def save_api_keys():
    with open(API_KEYS_FILE, 'w') as file:
        json.dump(api_keys, file)

#----------------------------------------------------------------
# Save API usage data to the file
def save_api_usage():
    with open(API_USAGE_FILE, 'w') as file:
        json.dump(api_usage, file)


#----------------------------------------------------------------
def log_api_usage(key, function_name, user_inputs, model_outputs):
     # Update call metadata
    api_keys[key]['total_calls'] += 1
    save_api_keys()

    # Log call metadata (you can customize this according to your requirements)
    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'api_key': key,
        'function': f'{function_name}',
        'arguments': user_inputs,
        'result': model_outputs
    }
    
    api_usage[key]['usage'].append(log_data)
    print(log_data)
    save_api_keys()
    

# Example function that can be called via the API
#----------------------------------------------------------------
def add_numbers(a, b):
    return a + b
    

#----------------------------------------------------------------
# Generates a new API key for a user
def generate_api_key():
    key = str(uuid.uuid4())
    api_keys[key] = {
        'created_at': datetime.datetime.now().isoformat(),
        'total_calls': 0,
        'usage' : []
    }
    save_api_keys()
    return key
    

#----------------------------------------------------------------
# Authenticates an API call using the provided key
def authenticate(key):
    return key in api_keys


#----------------------------------------------------------------
# API endpoint for generating a new API key
@app.route('/api/solve_quickbit', methods=['POST','GET'])
def solve_quickbit():
    data = request.get_json()
    key = data['api_key']
    if not authenticate(key):
        return jsonify({'error': 'Invalid API key'})

    print(f"{data}")
    data.pop('api_key')

    try:
        qb = Blank()
        qb.__dict__.update(data)
        qb.__class__ = QuickBit
        #qb = QuickBit.__new__(QuickBit)
        qb.__dict__.update(data)
        
        input_sequence = qb.build_header().decode('utf-8')
        print(f"{input_sequence       = }")
        print(f"{input_sequence[:-36] = }")
        preblock_header = DeployModel(model, input_sequence)
        qb.reset_from_preblockheader(preblock_header)
        api_generation = qb.parameters()
        api_generation.update({'calc_hash', qb.calculate_header_mining_hash()})
        log_api_usage(key, 'solve_quickbit', data, api_generation)
        return jsonify(api_generation)
        
    except:
        erm = f"Invalid json data. Retry sending data using template -> data = {'ver':int, 'prev_hash':str, 'mrkl_root':str, 'time':int, 'bits':int, 'nonce':int}.  Block-chain info can be found on https://www.blockchain.com/explorer  or  using the api  https://blockchain.info/rawblock/$block_hash"
        return jsonify({'error': erm})


#----------------------------------------------------------------
# API endpoint for generating a new API key
@app.route('/api/generate_key', methods=['POST'])
def api_generate_key():
    data = request.get_json()
    key = generate_api_key()
    api_keys[key].update({'username':data['username']})
    return jsonify({'api_key': key})


#----------------------------------------------------------------
# API endpoint for calling the add_numbers function
@app.route('/api/add', methods=['POST'])
def api_add():
    data = request.get_json()
    key = data['api_key']
    if not authenticate(key):
        return jsonify({'error': 'Invalid API key'})

    a = data['a']
    b = data['b']
    result = add_numbers(a, b)

    # Update call metadata
    api_keys[key]['total_calls'] += 1
    save_api_keys()

    # Log call metadata (you can customize this according to your requirements)
    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'api_key': key,
        'function': 'add_numbers',
        'arguments': {'a': a, 'b': b},
        'result': result
    }
    print(log_data)

    return jsonify({'result': result})


#----------------------------------------------------------------
#----------------------------------------------------------------
if __name__ == '__main__' and __package__ is None:
    __package__ = "QuickBitsNN.api"
    global model
    load_api_keys()
    load_api_usage()
    checkpoint = CheckPoint()
    checkpoint.label = "📏"
    loaded_checkpoint = LoadSavedCheckpoint(f'{filename}')
    model = load_model(loaded_checkpoint)
    model.to(get_device())

    if get_device() == 'cuda': app.run(host="0.0.0.0", port=5000)
    if get_device() == 'mps': app.run(host="0.0.0.0", port=8080)

#----------------------------------------------------------------
#----------------------------------------------------------------




#----------------------------------------------------------------
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


