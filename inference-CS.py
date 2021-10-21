import math, pickle, os, glob
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
import os.path
import sys5_lang as M
import librosa
import soundfile as sf
import operator
from functools import reduce
from collections import defaultdict

def load_wav(filename, encode=False):
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : 
        x = encode_16bits(x)
    return x

import config

num_speakers = 31

test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_siwis_lang.43.upconv_552024.pyt"

if __name__ == "__main__":

    model_path = test_model
    print("loading",model_path)
    sample_rate = config.sample_rate
    #test_model = model_path.split("_")[-1].split(".")[0]​
    test_model = model_path.split("_")[-1].split(".")[0]

    model = M.Model(4, nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()

    model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    pad_left = model.pad_left()
    pad_left_encoder = model.pad_left_encoder()
    pad_left_decoder = model.pad_left_decoder()
    pad_right = model.pad_right()
    window = 16
    
    # create the output directory

    output_dir = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inferenced/"
    os.makedirs(output_dir, exist_ok=True)

    # get the list of input files
    source_dir = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inference_sample_wav/"
    source_files = glob.glob(source_dir+"/*.wav")
    source_files = [s.split("/")[-1].split(".")[0] for s in source_files]
    print(source_files)

    for k, source in enumerate(source_files):
        source_filelist = [source]
        lang1 = source.split("_")[7]
        lang2 = source.split("_")[11]
        target_lang_codes = [lang1]
        print(target_lang_codes)
        path = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inference_sample_wav/"
        source_wav_files = [load_wav(f'{path}/{item}.wav') for item in source_filelist]
        source_name = [item.split(".")[0].split("/")[-1] for item in source_filelist]
        source_audio_files = [(wav * (2**15 - 0.5) - 0.5).astype(np.int16) for wav in source_wav_files]
        max_offsets = [x.shape[-1] - model.win for x in source_audio_files]
        offsets = [np.random.randint(0, offset) for offset in max_offsets]
        wave16 = [np.concatenate([np.zeros(model.left, dtype=np.int16), x, np.zeros(model.right, dtype=np.int16)])[offsets[i]:offsets[i] + model.left + model.win + model.right] for i, x in enumerate(source_audio_files)]
        source_wave16 = torch.LongTensor(np.stack(wave16).astype(np.int64)).cuda()
        source_total_f = (source_wave16.float() + 0.5) / 32767.5
        source_translated = source_total_f[:, pad_left-pad_left_encoder:]
        source_n_points = len(source_audio_files)
        source_gt = [(x.astype(np.float32) + 0.5) / (2**15 - 0.5) for x in source_audio_files]
        source_extended = [np.concatenate([np.zeros(model.pad_left_encoder(), dtype=np.float32), x, np.zeros(model.pad_right(), dtype=np.float32)]) for x in source_gt]
        source_maxlen = max([len(x) for x in source_extended])
        source_aligned = [torch.cat([torch.FloatTensor(x).cuda(), torch.zeros(source_maxlen-len(x)).cuda()]) for x in source_extended]
        source_samples = torch.stack(source_aligned, dim=0).cuda()

        for i, x in enumerate(source_gt):
            for target_lang in target_lang_codes:
                lang = [target_lang]
                langDict = {"EN":0, "FR":1, "IT":2, "DE":3}
                source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
                source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
                source_LANG = torch.FloatTensor(source_l_onehot).cuda() 
                n = source_name[i]
                x = source_gt[i]
                fname1 = f'{output_dir}/{n}--original.wav'
                fname2 = f'{output_dir}/{n}--{target_lang}--cs.wav'

                # continue making files, timed out

                if not os.path.isfile(fname1) or not os.path.isfile(fname2):
                    out = model.forward_gen(source_samples, source_translated, source_LANG)
                    audio = out[0][:len(x)].cpu().numpy()
                    sf.write(f'{output_dir}/{n}--original.wav', x, sample_rate)
                    sf.write(f'{output_dir}/{n}--{target_lang}--cs.wav', audio, sample_rate)
    sys.exit()

    done_speaker = []

#    out = model.forward_spk_change2(source_samples, source_translated, source_samples, source_translated, source_LANG)

#    n = source_name[0]

#    x = source_gt[0]

#    audio = out[0][:len(x)].cpu().numpy()

#    sf.write(f'{output_dir}/{n}--original.wav', x, sample_rate)

#    sf.write(f'{output_dir}/{n}--vocoded.wav', audio, sample_rate)

#    source_spk = 

    for i,f in enumerate(vc_filelist):
        vc_spk = f.split("_")[0]
        target_filelist = [f]
        target_wav_files = [load_wav(f'{path}/{item}.wav') for item in target_filelist]
        target_name = [item.split(".")[0].split("/")[-1] for item in target_filelist]
        target_audio_files = [(wav * (2**15 - 0.5) - 0.5).astype(np.int16) for wav in target_wav_files]

        max_offsets = [x.shape[-1] - model.win for x in target_audio_files]

        offsets = [np.random.randint(0, offset) for offset in max_offsets]

        wave16 = [np.concatenate([np.zeros(model.left, dtype=np.int16), x, np.zeros(model.right, dtype=np.int16)])[offsets[i]:offsets[i] + model.left + model.win + model.right] for i, x in enumerate(target_audio_files)]

        target_wave16 = torch.LongTensor(np.stack(wave16).astype(np.int64)).cuda()

        target_total_f = (target_wave16.float() + 0.5) / 32767.5

        target_translated = target_total_f[:, pad_left-pad_left_encoder:]

        target_n_points = len(target_audio_files)

        target_gt = [(x.astype(np.float32) + 0.5) / (2**15 - 0.5) for x in target_audio_files]

        target_extended = [np.concatenate([np.zeros(model.pad_left_encoder(), dtype=np.float32), x, np.zeros(model.pad_right(), dtype=np.float32)]) for x in target_gt]

        target_maxlen = max([len(x) for x in target_extended])

        target_aligned = [torch.cat([torch.FloatTensor(x).cuda(), torch.zeros(target_maxlen-len(x)).cuda()]) for x in target_extended]

        target_samples = torch.stack(target_aligned, dim=0).cuda()

        target_langs = [l.split("_")[0] for l in target_filelist]

        target_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in target_langs]

        target_l_onehot = torch.FloatTensor([x for x in target_l_onehot])

        target_LANG = torch.FloatTensor(target_l_onehot).cuda()

        vc_speaker = "_".join(f.split("_")[:3])

        if vc_speaker in done_speaker:

            continue

        out = model.forward_spk_change2(source_samples, source_translated, target_samples, target_translated, target_LANG)

        print(out.shape)

        for j, x in enumerate(source_gt):

            n = source_name[j]

            audio = out[j][:len(x)].cpu().numpy()

            print(f, "audio", audio.shape)

            sf.write(f'{output_dir}/{n}--vc--{f}.wav', audio, sample_rate)

        done_speaker.append(vc_speaker)


