import math, pickle, os
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import *
import sys
import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess

import models.sys5 as M
import librosa
import operator
from functools import reduce
from collections import defaultdict
from operator import itemgetter
import glob
from scipy.io.wavfile import write
from pydub import AudioSegment

# functions related to handling textgrid files output by montreal forced aligner
import textgrid
import sys, random
import configs.config5_vctk as config
dat = "vctk"
num_speakers = 100
model_path = "checkpoints/vctk_models/sys5_vctk.43.upconv_753011.pyt"
pos = sys.argv[1]
mask = sys.argv[2]


def load_codes(f):
    input = open(f, "r")
    data = input.read().split("\n")[0]
    input.close()
    l = data.split(" ")
    codes = [int(value) for value in l if value != '']
    return codes


def process_textgrid(tg):
    """
    extract phone and word alignments from textgrid file
    """
    tg = textgrid.TextGrid.fromFile(tg)
    phones, words = [], []
    words_intervaltier = tg[0]
    phones_intervaltier = tg[1]
    for word in words_intervaltier:
        # print(bool(word.mark))
        words.append({
            "word": word.mark if word.mark else 'SILENCE',
            "start": word.minTime,
            "end": word.maxTime,
        })
    for phone in phones_intervaltier:
        # print(bool(phone.mark))
        phones.append({
            "phone": phone.mark,
            "start": phone.minTime,
            "end": phone.maxTime,
        })
    return phones, words


def process_phrase(target, words):
    T = target.split(" ")
    first_word = T[0]
    last_word = T[-1]
    result = []
    for i, w in enumerate(words):
        if w["word"] == first_word:
            result.append(w["start"])
        if w["word"] == last_word:
            result.append(w["end"])
    end = words[-1]["end"]
    result.append(end)
    return result


def load_wav(filename, encode=False) :
    print(filename)
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : x = encode_16bits(x)
    return x


def save_codes(f, codes):
    C = []
    for c in codes:
        C.append(str(c[0]))
    output = open(f, "w")
    outstring = " ".join(C)
    output.write(outstring)
    output.close()
    return 


def get_codes(NAME, i):
    name = [NAME[i]]
    speaker_id = [NAME[i].split("_")[0]]
    wav_files = [load_wav(f'{data_path}/{speaker_id[0]}/{n}.wav') for spk, n in zip(speaker_id, name)]  
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
    phn_codes, phn_vecs, spk_codes, spk_vecs, all_phn, all_spk = model.forward_codedump(unpadded, padded)
            
    phn_out = phn_codes.data.cpu().numpy()
    spk_out = spk_codes.data.cpu().numpy()
    vq_codes, spk_code = [], []
    for c in phn_out:
        vq_codes.append(c[0])
    for c in spk_out:
        spk_code.append(c[0])
        
    n = name[0]
    print("extracted codes", n, spk_code)
        
    tgfile = f'/home/s1738075/data/VCTK-0.92/vctk_montreal_alignments_from_trimmed_wavs/{speaker_id[0]}/{n}.TextGrid'
    phones, words = process_textgrid(tgfile)
    return words, phones, vq_codes, spk_code


W = defaultdict(int)
WR = defaultdict(int)
P = defaultdict(int)

print("loading",model_path)
sample_rate = config.sample_rate
model = M.Model(nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
pad_left = model.pad_left()
pad_left_encoder = model.pad_left_encoder()
pad_left_decoder = model.pad_left_decoder()
pad_right = model.pad_right()
window = 16

input = open("test_lists/spsc_natural.txt", "r")
data = input.read().split("\n")[:-1]
input.close()
SID = [item.split(",")[1] for item in data]
NAME = [item.split(",")[0] for item in data]
targets = [2, 3, 4]
data_path = "/home/s1738075/data/VCTK-0.92/vqvae_wavernn_trimmed_wavs"

noise = AudioSegment.from_wav("00_folder/noise_16k.wav")
silence = AudioSegment.from_wav("00_folder/silence_16k.wav")



for i in range(0, len(NAME)):
    name = [NAME[i]]
    n = name[0]

    name = [NAME[i]]
    speaker_id = [NAME[i].split("_")[0]]
    wav_file = f'{data_path}/{speaker_id[0]}/{n}.wav'
    speech_file = AudioSegment.from_wav(wav_file)

    tgfile = f'/home/s1738075/data/VCTK-0.92/vctk_montreal_alignments_from_trimmed_wavs/{speaker_id[0]}/{n}.TextGrid'
    phones, words = process_textgrid(tgfile)

    num_words = len(words)
    num_phones = len(phones)
    word_rate = int(num_words / words[-1]["end"])
    W[num_words] += 1
    P[num_phones] += 1
    WR[word_rate] += 1
    total_dur = words[-1]["end"]
    
    if word_rate < 5 and num_words >= 7:
       
        for i, percent in enumerate(targets):
            # determine location in code sequence to do the splice
            if pos == "start":
                word_string = words[:percent]
                word_start = word_string[0]["start"]
                word_end = word_string[-1]["end"]
                loc1 = 0
                mask_time = word_end - word_start
                mask_time_ms = mask_time*1000
                loc2 = mask_time_ms
                print("loc1", loc1, "loc2", loc2, percent, mask_time_ms, "start")
                
            if pos == "mid":
                freewords = len(words) - percent
                splitspace = int(freewords / 2)                
                word_string = words[splitspace:(splitspace+percent)]
                word_start = word_string[0]["start"]
                word_end = word_string[-1]["end"]
                mask_time = word_end - word_start
                mask_time_ms = mask_time*1000
                loc1 = word_start * 1000
                loc2 = loc1 + mask_time_ms
                print("loc1", loc1, "loc2", loc2, percent, mask_time_ms, "mid")
                
            if pos == "end":
                word_string = words[-percent:]
                word_start = word_string[0]["start"]
                word_end = word_string[-1]["end"]
                mask_time = word_end - word_start
                mask_time_ms = mask_time*1000
                loc1 = (word_end*1000) - mask_time_ms
                loc2 = word_end*1000
                print("loc1", loc1, "loc2", loc2, percent, mask_time_ms, "end")
                
                
            # replace vq codes with noise codes
            noise_sequence = noise[0:mask_time_ms]
            silence_sequence = silence[0:mask_time_ms]
            

            
            rev = speech_file.reverse()
            l1 = (total_dur*1000)-loc2
            l2 = l1 + mask_time_ms
            ss0 = rev[l1:l2]
            
            ss1 = speech_file[:loc1]
            ss2 = speech_file[loc2:]

            
            ss3_noise = ss1 + noise_sequence + ss2
            ss3_silence = ss1 + silence_sequence + ss2
            ss3_reverse = ss1 + ss0 + ss2

            if mask == "noise":
                ss3 = ss3_noise
            if mask == "silence":
                ss3 = ss3_silence
            if mask == "reversal":
                ss3 = ss3_reverse


            outdir = "/home/s1738075/special/inference_mask/"+pos+"_"+mask+"_"+str(percent)+"_natural/"
            os.makedirs(outdir, exist_ok=True)
            fname = outdir+f'/{n}_generated.wav'
            ss3.export(fname, format="wav")

