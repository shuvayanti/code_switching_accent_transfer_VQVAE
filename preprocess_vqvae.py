# ==================================================================================================
# Copyright (c) 2021, Jennifer Williams and Yamagishi Laboratory, National Institute of Informatics
# Author: Jennifer Williams (j.williams@ed.ac.uk)
# All rights reserved.
# ==================================================================================================


import sys
import glob
import pickle
import os
import multiprocessing as mp
#from utils.dsp import *

import sys
import glob
import pickle
import os
import multiprocessing as mp
#from utils.dsp import *

SEG_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"     #"/path/to/normalised_wavs"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_output_updated/"       #"/path/to/speaker/output"
SPKIND_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/"      #"/path/to/speaker/index"

def get_files(path):
    next_speaker_id = 0
    speaker_ids = {}
    filenames = []
    for filename in glob.iglob(f'{path}/*.wav', recursive=True):
        speaker_name = filename.split('/')[-1].split("_")[2]
        if speaker_name not in speaker_ids:
            speaker_ids[speaker_name] = next_speaker_id
            next_speaker_id += 1
            filenames.append([])
        filenames[speaker_ids[speaker_name]].append(filename)

    return filenames, speaker_ids

def process_file(i, path):
    dir = f'{DATA_PATH}/'
    name = path.split('/')[-1][:-4] # Drop .wav
    #print('name of the wav file:',name)
    filename = f'{dir}/{name}.npy'
    if os.path.exists(filename):
        #print(f'{filename} already exists, skipping')
        pass
    floats = load_wav(path, encode=False)
    trimmed, _ = librosa.effects.trim(floats, top_db=25)
    quant = (trimmed * (2**15 - 0.5) - 0.5).astype(np.int16)
    if max(abs(quant)) < 2048:
        print(f'audio fragment too quiet ({max(abs(quant))}), skipping: {path}')
        return
    if len(quant) < 10000:
        print(f'audio fragment too short ({len(quant)} samples), skipping: {path}')
        return
    os.makedirs(dir, exist_ok=True)
    np.save(filename, quant)
    return name

files, spks = get_files(SEG_PATH)
index = []
with mp.Pool(8) as pool:
    for i, speaker in enumerate(files):
        #print("starting speaker: ", speaker)
        try:
            res = pool.starmap_async(process_file, [(i, path) for path in speaker]).get()
        except:
            continue
        print("got res:",res)
        index.append([x for x in res if x])
        print("index = ", index)
        print(f'Done processing speaker {i}')

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(SPKIND_PATH, exist_ok=True)
with open(f'{SPKIND_PATH}/index.pkl', 'wb') as f:
    pickle.dump(index, f)


spk_map = "/home/s1995633/s1995633/dissertation/siwis_database/spkmap_siwis_updated.txt"        #"/path/to/speaker/map"
output = open(spk_map, "w")
for k,v in spks.items():
    output.write(str(k)+","+str(v)+"\n")
output.close()
