import json
import hashlib
import tempfile
import shutil
import sys, os, os.path

from ..data_io.const import DATA_FILE_PATH, INCOMPLETE_DATA_FILE_PATH
from ..framework.quickbit import QuickBit, save_quickbits, load_quickbits



#----------------------------------------------------------------
#----------------------------------------------------------------
def find_files_with_similar_names(folder_path: str, filename_prefix: str):
    matching_files = []

    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file name starts with the given prefix
        if file_name.startswith(filename_prefix):
            matching_files.append(file_name)

    return matching_files



#----------------------------------------------------------------
def append_to_json_file(file_path: str, new_data: list):

    try:
        temp_file_path = tempfile.mktemp()  # Create a temporary file

        # Read existing data from the JSON file in chunks and write it to the temporary file
        with open(file_path, 'r') as read_file, open(temp_file_path, 'w') as temp_file:
            temp_file.write('[')  # Start the JSON array in the temporary file

            # Skip the opening '[' character in the existing file
            read_file.seek(1)

            for line in read_file:
                temp_file.write(line)

            temp_file.seek(0, os.SEEK_END)  # Move the file pointer to the end
            temp_file.seek(temp_file.tell() - 2)  # Move the file pointer before the closing ']' character
            temp_file.write(',')  # Add a comma to separate the existing and new data

        # Append the new data to the temporary file
        with open(temp_file_path, 'a', encoding="utf-8") as temp_file:
            for i, data in enumerate(new_data):
                json.dump(data, temp_file, indent=4, default=lambda o: o.__dict__)
                if i < len(new_data) - 1:
                    temp_file.write(',')  # Add a comma to separate the new data dictionaries

            temp_file.write(']')  # End the JSON array in the temporary file

        # Replace the original file with the temporary file
        shutil.move(temp_file_path, file_path)
    except:
        save_quickbits(new_data, file_path)


#----------------------------------------------------------------
def sort_key(entry):
    leading_zeros = len(entry.hash) - len(entry.hash.lstrip('0'))
    return leading_zeros, entry.nonce


#----------------------------------------------------------------
def AppendQuickBitsFile(data, filename):

    try:
        prev_quickbits = load_quickbits(filename)
        prev_quickbits.extend(data)
        print(f'Total quickbits for file: {len(prev_quickbits)}')
        save_quickbits(prev_quickbits, filename)
        prev_quickbits.clear()
    except:
        print('No file to append')
        save_quickbits(data, filename)

def append_quickbits_file(data, filename):

    try:
        prev_quickbits = load_quickbits(filename)
        prev_quickbits.extend(data)
        print(f'Total quickbits for file: {len(prev_quickbits)}')
        save_quickbits(prev_quickbits, filename)
        prev_quickbits.clear()
    except:
        print('No file to append')
        save_quickbits(data, filename)



#----------------------------------------------------------------
def SortQuickBits( quickbits ):

    print(f"QuickBits to Sort: {len(quickbits)}.")
    sorted_data = sorted(quickbits, key=sort_key)

    bins = {}
    for entry in sorted_data:
        leading_zeros = len(entry.hash) - len(entry.hash.lstrip('0'))
        if leading_zeros not in bins:
            bins[leading_zeros] = []
        bins[leading_zeros].append(entry)


    print(f"Total Bins: {len(bins)}.")
    [print(f"Leading Zeros: {bin: >10}  Items: {len(entries): >8}") for bin,entries in bins.items() ]

    for leading_zeros, entries in bins.items():
        if leading_zeros == 14:
            filename = INCOMPLETE_DATA_FILE_PATH + f'quickbins_({leading_zeros})0s.json'

            #append_quickbits_file(entries, filename)
            append_to_json_file(filename, entries)
            
            

#----------------------------------------------------------------
def build_preblock_header_lookup_file(all_quickbits):
    hash_lookup = {}
    for bit in all_quickbits:
        hash_lookup.update({bit.hash : f"{bit.build_header().decode('utf-8')}"})
        
    save_json_file(hash_lookup, INCOMPLETE_DATA_FILE_PATH + f'quickbits_preblock_headers.json')
    
