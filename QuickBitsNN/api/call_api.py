import requests
import json

#----------------------------------------------------------------
#----------------------------------------------------------------
def GenerateKey(api_url=f'http://localhost:5000/api/generate-key'):

    response = requests.post(api_url, json=data)
    result = response.json()['api_key']
    print(result)


#----------------------------------------------------------------
def SolveNonceFromHeader(data, api_url='http://localhost:5000/api/solve_quickbit'):

    response = requests.post(api_url, json=data)
    result = response.json()
    print(result)


#----------------------------------------------------------------
def test_data(type=f'quickbit'):

    #  Example on Generating API Key
    if type == 'key_gen':
        api_url=f'http://localhost:5000/api/generate-key'
        data = { 'username' : '); DROP TABLE Users;--' }

    # For Testing API functionality
    if type == 'add':
        api_url='http://localhost:5000/api/add'
        data = {
            'api_key' : '4582a541-a25f-4471-ab05-1f00f50635b4',
            'a' : 5,
            'b' : 3
        }
    return data, api_url

    # Example of QuickBit Data to Submit to API
    if type == 'quickbit':
        api_url='http://localhost:5000/api/solve_quickbit'
        data = {'api_key' : 'a9256839-574c-46de-8dfc-4b7b4396cf08',
            'ver': 549453824,
            'prev_block':'00000000000000000001a0dc3ad008662971ffddbf06cfa6c8c67cdc95777b6d',
            'mrkl_root':'6652d1df6a7e7c94fbe6d264a114440e2c2b8ed5c6c91a5c29b3421b74477f01',
            'time': 1688154065,
            'bits': 386240190,
            'nonce': 3582160702}
    return data, api_url
    
#----------------------------------------------------------------
#----------------------------------------------------------------

