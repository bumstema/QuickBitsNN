import json
from datetime import datetime, timedelta, time, date

from ..data_io.const import DATA_FILE_PATH, INCOMPLETE_DATA_FILE_PATH
from ..framework.quickbit import  QuickBit, save_quickbits, load_quickbits, preprocess_quickbit_json


#----------------------------------------------------------------
#----------------------------------------------------------------
def ping_local_core(command, hash):
    result = subprocess.run(['bitcoin-cli', command, hash], capture_output=True, text=True )
    if result.returncode == 0:
        response = json.loads(result.stdout)
        data = preprocess_quickbit_json(response)
        return data
    else:
        print(f'{result}')
        print('error')
        return None
        
        
#----------------------------------------------------------------
def QuickBitFromCore(hash):
    command = f'getblock {hash}'
    data = ping_local_core(command)
    data = [preprocess_quickbit_json(item) for item in data]
    return [QuickBit(**quickbit) for quickbit in data]


#----------------------------------------------------------------
def SniffQuickBits( resume_quickbit_hash, n_quickbits=100000, direction='forward', batchsniff=1):
    print(f'Now Sniffing QuickBits...')

    first_quickbit =  ping_local_core(f'getblock', f'{resume_quickbit_hash}')
    first_quickbit = QuickBit( **first_quickbit )
    print(f"Sniffing in the: _{direction}_ Direction.")

    if direction == 'forward' :
        next_quickbit = first_quickbit.next_block[0]
        quickbits = []
        i  = 0

    if direction == 'backward':
        next_quickbit = first_quickbit.prev_block
        quickbits = [first_quickbit]
        i  = 1
        print(f"First QuickBit Hash: {quickbits[0].hash}")


    total_loops = 0
    TOTAL_QUICKBITS = n_quickbits
    start_time = datetime.now()

    try:
        while len(quickbits) < TOTAL_QUICKBITS :

            quickbit_data = ping_local_core(f'getblock',f'{next_quickbit}')

            temp_quickbit = QuickBit(**quickbit_data)
            quickbits.append(temp_quickbit)

            i += 1
            total_loops += 1

            if i % (TOTAL_QUICKBITS/10) == 0:
                print(f'...sniffed QuickBits: {i}/{TOTAL_QUICKBITS} out of ({total_loops}) loopbits [Time = {str(datetime.now() - start_time).split(".", 2)[0]}]...')

            # Sniff Forward in Time from Current QuickBit
            if direction == 'forward': next_quickbit = temp_quickbit.next_block[0]
            # Sniff Backward in Time from Current QuickBit
            if direction == 'backward': next_quickbit = temp_quickbit.prev_block

            
        print(f'DONE! Total of Quickbits: ({len(quickbits)}) fully sniffed out of: ({total_loops}). [Time = {str(datetime.now() - start_time).split(".", 2)[0]}]')
        save_quickbits(quickbits,  INCOMPLETE_DATA_FILE_PATH + f'quickbits_validation_coresniff_{direction}_{batchsniff}_({len(quickbits)}).json')
        print(f"Last QuickBit Hash: {quickbits[-1].hash}  QuickBit Time: {quickbits[-1].time}")

        return quickbits

    except:
        print(f'Error occured at ({len(quickbits)}) quickbits')
        save_quickbits(quickbits, INCOMPLETE_DATA_FILE_PATH + f'quickbits_validation_coresniff_{direction}_{batchsniff}_({len(quickbits)}).json')
        print(f"Error QuickBit Hash: {quickbits[-1].hash}")

    print(f"Done! Last QuickBit Hash: {quickbits[-1].hash}  QuickBit Time: {quickbits[-1].time}")



#----------------------------------------------------------------
def main():
    genesis_block = f'000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f'
    #print(f'Genesis Block: {genesis_block}')
    quickbits = SniffQuickBits(genesis_block)
    print(f"{quickbits[-1].hash}")
    save_quickbits(quickbits, DATA_FILE_PATH)



#----------------------------------------------------------------
#----------------------------------------------------------------
if __name__ == '__main__':
    main()






