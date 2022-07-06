import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from utils import *
import sys
#import utils.env as env
import argparse
import platform
import re
#import utils.logger as logger
import time
import subprocess
import modules.sys5_lang as M
import librosa
import operator
from functools import reduce
from collections import defaultdict

print(torch.__version__)
import glob
#import configs.config5_siwis as config
import modules.config
num_speakers = 31 # siwis model
#num_speakers = 100 #VCTK model
lang_dims = 4
dat = "siwis"
test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_siwis_lang.43.upconv_552024.pyt"   #"/path/to/siwis/pretrained/model"
#test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_vctk.43.upconv_753011.pyt"
condition = "train"
#langDict = {"EN":0, "FR":1, "IT":2, "DE":3}​
langDict = {"EN":0, "FR":1, "IT":2, "DE":3}
def load_wav(filename, encode=False) :
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : x = encode_16bits(x)
    return x

def load_text(path):
    input = open(path, "r")
    data = input.read().split("\n")[0]
    input.close()
    return data

if __name__ == "__main__":
    model_path = test_model
    print("loading",model_path)
    sample_rate = config.sample_rate
    test_model = "VCTK_"+model_path.split("_")[-1].split(".")[0]
    print(test_model)
    #model = M.Model(lang_dims=lang_dims, nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    model = M.Model(nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    pad_left = model.pad_left()
    pad_left_encoder = model.pad_left_encoder()
    pad_left_decoder = model.pad_left_decoder()
    pad_right = model.pad_right()
    window = 16
    
    if dat == "vctk":
        data_path = "/home/s1738075/data/VCTK-0.92/wav16_trimmed"
        with open("/home/s1738075/data/VCTK-0.92/wav16_trimmed/index.pkl", 'rb') as f:
            index = pickle.load(f)
        train_set = index[:-10]
        unseen_set = index[-10:]
        test_index = [x[:10] for i, x in enumerate(train_set)]
        train_index = [x[10:-1] if i < 1 else x for i, x in enumerate(train_set)]
        if condition == "train":
            SID = [item[0].split("_")[0] for item in train_index]
            NAME = [name for item in train_index for name in item]
        elif condition == "test":
            print(test_index)
            SID = [item[0].split("_")[0] for item in test_index]
            NAME = [name for item in test_index for name in item]
        else:
            input = open("test_lists/"+condition+".txt", "r")
            data = input.read().split("\n")
            input.close()
            SID = [item.split(",")[1] for item in data]
            NAME = [item.split(",")[0] for item in data]
        COMPRESSION = []
    if dat == "siwis":
        data_path = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"                    #"/path/to/siwis/normalised/wavs"
        with open("/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl", 'rb') as f:    #"/path/to/siwis/speaker_index"
            index = pickle.load(f)
        train_set = index[:]
        #print(train_set)
        test_index = [x[:75] for i, x in enumerate(train_set)]
        train_index = [x[75:] for i, x in enumerate(train_set)]
        print("Train set speakers: ", len(train_index))
        print("Train set total: ", sum( [ len(listElem) for listElem in train_index]))
        print("Test set speakers: ", len(test_index))
        print("Test set total: ", sum( [ len(listElem) for listElem in test_index]))
        if condition == "train":
            SID = [item[0].split("_")[2] for item in train_index]
            #NAME = [[name for item in train_index] for name in item]
            #NAME = [[name for name in item] for item in train_index]
            NAME=[]
            for item in train_index:
                for name in item:
                    NAME.append(name)
        elif condition == "test":
            print(test_index)
            SID = [item[0].split("_")[2] for item in test_index]
            NAME = [name for item in test_index for name in item]
        else:
            input = open("test_lists/"+condition+".txt", "r")
            data = input.read().split("\n")[:-1]
            input.close()
            SID = [item[2] for item in data]
            NAME = [item for item in data]
        COMPRESSION = []
        
    
    for i in range(0, len(NAME)):
        try:
            if dat == "vctk":
                name = [NAME[i]]
                speaker_id = [NAME[i].split("_")[0]]
                langs = ["EN"] * len(name)
                condition = condition+"_vctk"
            if dat == "siwis":
                name = [NAME[i]]
                speaker_id = [NAME[i].split("_")[0]]
                langs = [NAME[i].split("_")[0]]
                
            wav_files = [load_wav(f'{data_path}/{n}.wav') for spk, n in zip(speaker_id, name)]
            audio_files = [(wav * (2**15 - 0.5) - 0.5).astype(np.int16) for wav in wav_files]
            max_offsets = [x.shape[-1] - model.win for x in audio_files]
            offsets = [np.random.randint(0, offset) for offset in max_offsets]
            wave16 = [np.concatenate([np.zeros(model.left, dtype=np.int16), x, np.zeros(model.right, dtype=np.int16)])[offsets[i]:offsets[i] + model.left + model.win + model.right] for i, x in enumerate(audio_files)]       
            wave16 = torch.LongTensor(np.stack(wave16).astype(np.int64)).cuda()
            total_f = (wave16.float() + 0.5) / 32767.5
            padded = total_f[:, pad_left-pad_left_encoder:]
            
            wave16 = torch.LongTensor(np.stack(audio_files).astype(np.int64)).cuda()
            total_f = (wave16.float() + 0.5) / 32767.5
            unpadded = total_f[:, :]
            LANG = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in langs]
            l_onehot = torch.FloatTensor([x for x in LANG])
            L = torch.FloatTensor(l_onehot).cuda()

#            phn_codes, phn_vecs = model.forward_codedump(unpadded, padded)
#            phn_codes, phn_vecs, spk_codes, spk_vecs = model.forward_codedump(unpadded, padded)
            print("dumping codes")
            phn_codes, phn_vecs, spk_codes, spk_vecs, all_phn, all_spk = model.forward_codedump(unpadded, padded)
            print("finished dumping codes\n now moving on to storing")
#            np.save("vq_codebooks/sys5.vctk.vq_phn_embs.npy", all_phn.data.cpu().numpy())
#            np.save("vq_codebooks/sys5.vctk.vq_spk_embs.npy", all_spk.data.cpu().numpy())
            #sys.exit()
            
            n = NAME[i]
#            unpadded = unpadded.data.cpu().numpy()
#            unpadded_dir = "vq_unpadded_data/sys5_lang/"+test_model+"/"+condition
#            os.makedirs(unpadded_dir, exist_ok=True)
#            np.save(unpadded_dir+"/"+n, unpadded)

            phn_codes = phn_codes.data.cpu().numpy()
            phn_code_dir = "vq_phn_codes_updated/sys5_lang/"+test_model+"/"+condition
            os.makedirs(phn_code_dir, exist_ok=True)
            np.save(phn_code_dir+"/"+n, phn_codes)
            phn_vecs = phn_vecs.data.cpu().numpy()[0]
            phn_vec_dir = "vq_phn_vecs_updated/sys5_lang/"+test_model+"/"+condition
            os.makedirs(phn_vec_dir, exist_ok=True)
            np.save(phn_vec_dir+"/"+n, phn_vecs)
            
            spk_codes = spk_codes.data.cpu().numpy()
            spk_code_dir = "vq_spk_codes_updated/sys5_lang/"+test_model+"/all_siwis"
            os.makedirs(spk_code_dir, exist_ok=True)
            np.save(spk_code_dir+"/"+n, spk_codes)
            spk_vecs = spk_vecs.data.cpu().numpy()[0]
            spk_vec_dir = "vq_spk_vecs_updated/sys5_lang/"+test_model+"/all_siwis"
            os.makedirs(spk_vec_dir, exist_ok=True)
            np.save(spk_vec_dir+"/"+n, spk_vecs)
            
            print("completed", n)
            
        except:
            continue
sys.exit()


print("num", len(COMPRESSION))
print("mean", np.mean(np.asarray(COMPRESSION)))
print("std",  np.std(np.asarray(COMPRESSION)))

plt.hist(COMPRESSION)
plt.title("VCTK Train Set Phone Encoder Compression (unpadded)")
#plt.xlabel("Compression")
#plt.ylabel("")
plt.locator_params(axis="x", integer=True, tight=True)
#plt.legend()
plt.savefig("VCTK_train_compression.png")
plt.clf()
