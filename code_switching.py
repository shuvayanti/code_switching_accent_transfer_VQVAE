import math, pickle, os, glob
import numpy as np
import torch
import random
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
import config
from functools import reduce
from collections import defaultdict
from operator import itemgetter
from scipy.io.wavfile import write
from pydub import AudioSegment
#from content_masking_natural import get_codes,load_codes

def load_wav(filename, encode=False):
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : 
        x = encode_16bits(x)
    return x

PHN_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5_lang/siwis_552024/train/"
PHN_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_vecs_updated/sys5_lang/siwis_552024/train/"
SPK_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_vecs_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"

num_speakers = 31
test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_siwis_lang.43.upconv_552024.pyt"

langDict = {"EN":0, "FR":1, "IT":2, "DE":3}

def load_phnfile(filename, encode = False):
	return np.load(open(PHN_CODE_PATH+filename,'rb'))

def load_spkfile(filename, encode=False):
    return np.load(open(SPK_CODE_PATH+filename,'rb'))

def combine_phn_codes(file1,file2):
    
    if len(file1) == 0:
        #print(np.load(open(file2,'rb')))
        return np.load(open(file2,'rb'))
    elif type(file1) == str:
        return np.concatenate( [np.load(open(file1,'rb')), np.load(open(file2,'rb'))] )
    return np.append(file1,np.load(open(file2,'rb')))

def write_wav_file(output_filename, sample_rate):
    sf.write(output_filename+'.wav', audio, sample_rate)

def voice_conversion(vq_code, source_LANG, speaker_code):
    out = model.forward_audio_from_only_codes(vq_code, source_LANG, speaker_code)
    audio = out[0].cpu().numpy()
    return audio

def alignment_breakdown(file_path):
    lines = open(file_path).read().strip().splitlines()
    
    word_list=[]
    time_segment = []
    for line in lines:
        line = line.strip().split('\t')
        word_list.append(line[0])
        time_segment.append([float(line[2]), float(line[3])])
    
    return word_list, time_segment

def word_phoneCode_map(word_list, time_segment, vq_code):
    
    #word_to_phoneCode = dict()
    code_segments = []
    #phones = list(word_phone_map.values())
    #phones = [ph for phone_list in phones for ph in phone_list]
    
    print('length of VQ code: ',len(vq_code))
    #words = list(word_phone_map.keys())
    
    print(time_segment[-1][-1])
    #print('number of phones without silence :', len(phones[1:]))
    codes_per_phone = len(vq_code)/(time_segment[-1][-1])
    
    print('codes per second :',codes_per_phone)

    for segment in time_segment:
        start = segment[0]*codes_per_phone
        difference = segment[1] - segment[0]
        end = start + difference * codes_per_phone
        code_segments.append([math.ceil(start), math.ceil(end)])
    
    return code_segments

def speaker_breakdown():
    speaker_language = dict()
    speaker_id_code = dict()
    speaker_file = dict()
    
    #speaker-code & file mapping
    for i,f in enumerate(os.listdir(SPK_CODE_PATH)):
        s = np.load(open(SPK_CODE_PATH+f,'rb'))
        try:
            speaker_file[s[0][0]].append((i,f))
        except:
            speaker_file[s[0][0]]=[(i,f)]
    #print(len(speaker_file))
    #print(speaker_file.keys())
    #speaker-id & langauge mapping
    with open(SPK_MAP, 'rb') as f:
        index = pickle.load(f)
        for i, spk_list in enumerate(index):
            for item in spk_list:
                #print(item[:2])
                #if item.startswith('EN'):
                    #print('english file:',i,item)
                try:
                        #print('doing union for english file')
                    speaker_language[item[:2]] = speaker_language[item[:2]].union({i})
                except:
                        #print('first time adding engalish data to dictionary')
                    speaker_language[item[:2]]={i}
                '''
                elif item.startswith('IT'):
                    #print('italian file:',i,item)
                    try:
                        #print('doing union for italian file')
                        speaker_language['IT'] = speaker_language['IT'].union({i})
                    except:
                        #print('first time adding italian data to dictionary')
                        speaker_language['IT']={i}
                '''
                for code,files in speaker_file.items():
                    files = [f[1].split('.')[0] for f in files]
                    if item in files:
                        #print(i,code)
                        speaker_id_code[i]=code
                    #print(i,code)
    #print(speaker_language)
    return speaker_id_code, speaker_file, speaker_language

def linguistic_unit_single_speaker(speaker_id_code , speaker_file , speaker_language, speaker_id, unit_size =1):
    
    print('\nsize of unit: ', unit_size)
    output_filename = ''
    speaker_code = [speaker_id_code[speaker_id]]

    output_dir_1 = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/combine-phn-codes/"
    os.makedirs(output_dir_1, exist_ok=True)

    output_dir_2 = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/combine-audio/"
    os.makedirs(output_dir_2, exist_ok=True)

    output_dir_3 = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inferenced/"
    os.makedirs(output_dir_3, exist_ok=True)

    language = ['EN','FR','DE','IT']
    
    #combining phone codes and audio segments of 2 files

    file_EN = random.choice(os.listdir(PHN_CODE_PATH))
    file_EN_path = PHN_CODE_PATH +file_EN
    output_file = file_EN
    
    print(output_file)
    pick = output_filename.split('_')[0]
    while pick == output_filename.split('_')[0]: 
        print(pick)
        pick = random.choice(language)

    file_IT = ''
    file_IT_path = PHN_CODE_PATH + file_IT
    
    while not(file_IT.startswith(pick)):                       # and os.path.exists(file_IT_path)):
        file_IT = random.choice(os.listdir(PHN_CODE_PATH))
        file_IT_path = PHN_CODE_PATH + file_IT
    
    print('file 1: ',file_EN)
    print('file 2: ',file_IT)
    vq_code_EN = []
    vq_code_IT = []  
    
        
    alignment_EN = ALIGN_PATH + file_EN.split('.')[0] +'.txt'    

    vq_code_EN = combine_phn_codes(vq_code_EN,file_EN_path)
    word_list_EN, time_segment_EN = alignment_breakdown(alignment_EN)

    alignment_IT = ALIGN_PATH + file_IT.split('.')[0] +'.txt'

    

    vq_code_IT = combine_phn_codes(vq_code_IT,file_IT_path)
    word_list_IT, time_segment_IT = alignment_breakdown(alignment_IT)

    #print('original 1 code info: ', len(vq_code_EN))
    #print('original 2 code info: ', len(vq_code_IT))
    
    code_segments_EN = word_phoneCode_map(word_list_EN, time_segment_EN, vq_code_EN)
    code_segments_IT = word_phoneCode_map(word_list_IT, time_segment_IT, vq_code_IT)
    
    print('1 code segments: ',code_segments_EN)
    print('2 code segments: ',code_segments_IT)
    
    print('1 time segments: ',time_segment_EN)
    print('2 time segments: ',time_segment_IT)
    
    combined_vq_code=[]
    
    #while len(combined_vq_code)==0:

    if unit_size < len(word_list_EN)-5 and unit_size < len(word_list_IT)-5:
        random_index_EN = random.choice(range(4,len(word_list_EN)-unit_size-1))
        random_index_IT = random.choice(range(4,len(word_list_IT)-unit_size-1))
        
        replacement_segment_EN = code_segments_EN[random_index_EN:random_index_EN+unit_size]
        replacement_segment_IT = code_segments_IT[random_index_IT:random_index_IT+unit_size]

        print('replacement segment 1:', replacement_segment_EN)
        print('replacement segment 2:', replacement_segment_IT)
    
        replace_time_EN = time_segment_EN[random_index_EN:random_index_EN+unit_size]
        replace_time_IT = time_segment_IT[random_index_IT:random_index_IT+unit_size]
        
        print('time segment 1:', replace_time_EN)
        print('time segment 2:', replace_time_IT)
    
        combined_vq_code = np.concatenate([ vq_code_EN[: replacement_segment_EN[0][0] ], \
                                            vq_code_IT[ replacement_segment_IT[0][0]: replacement_segment_IT[-1][-1] ], \
                                            vq_code_EN[replacement_segment_EN[-1][-1]: ] ])
    else:
        return       

    output_file = 'speaker_'+str(speaker_id)+'_unit_size_'+str(unit_size)+'_indices_'+str(random_index_EN)+'&'+str(random_index_IT)+'_'+file_EN.split('.')[0]+'_'+file_IT.split('.')[0]+'.wav'
    print('output file: ',output_file)  
    
    #English audio file
    lang = [file_EN.split('_')[0]]
    
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    #if len():
    print('VQ code length of 1st segment: ',len(vq_code_EN[: replacement_segment_EN[0][0] ]))
    audio_EN_segment_1 = voice_conversion(vq_code_EN[0 : replacement_segment_EN[0][0] ], source_LANG, speaker_code)

    print('VQ code length of 2nd segment: ',len(vq_code_EN[replacement_segment_EN[-1][-1]: ]))
    audio_EN_segment_2 = voice_conversion(vq_code_EN[replacement_segment_EN[-1][-1]: -1], source_LANG, speaker_code)

    audio_EN = voice_conversion(combined_vq_code, source_LANG, speaker_code)
    sf.write(output_dir_1+output_file, audio_EN, sample_rate)
    
    #Italian audio file
    lang = [file_IT.split('_')[0]]
    
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()

    audio_IT = voice_conversion(vq_code_IT[ replacement_segment_IT[0][0]: replacement_segment_IT[-1][-1] ], source_LANG, speaker_code)

    if len(audio_EN_segment_1) and len(audio_EN_segment_2):
        audio = np.concatenate( [ audio_EN_segment_1, audio_IT, audio_EN_segment_2] )
    elif len(audio_EN_segment_1):
        audio = np.concatenate( [ audio_EN_segment_1, audio_IT] )
    else:
        audio = np.concatenate( [audio_IT, audio_EN_segment_2] )
    sf.write(output_dir_2+output_file, audio, sample_rate)
    
    #generating audio from concatenated wav file and re-generating the audio from the codes extracted'
    audio_file_1 = AudioSegment.from_wav(DATA_PATH+file_EN.split('.')[0]+'.wav')
    audio_file_2 = AudioSegment.from_wav(DATA_PATH+file_IT.split('.')[0]+'.wav')

    new_audio = audio_file_1[:replace_time_EN[0][0]*1000] + audio_file_2[replace_time_IT[0][0]*1000:replace_time_IT[-1][-1]*1000]+ audio_file_1[replace_time_EN[-1][-1]*1000: ]
    print(new_audio)
    output_dir_4 = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inference_sample_wav/"
    os.makedirs(output_dir_4, exist_ok=True) 
    new_audio.export(output_dir_4+output_file, format="wav")
        
def generation_from_inference():
    audio_files = glob.glob("/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inference_sample_wav/*.wav")    
    audio_files = [s.split("/")[-1].split(".")[0] for s in audio_files]
    
    path = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inference_sample_wav/"
    wav_files = [load_wav(f'{path}/{item}.wav') for item in audio_files]  
    audio_files = [(wav * (2**15 - 0.5) - 0.5).astype(np.int16) for wav in wav_files]
    max_offsets = [x.shape[-1] - model.win for x in audio_files]
    offsets = [np.random.randint(0, offset) for offset in max_offsets]
    wave16 = [np.concatenate([np.zeros(model.left, dtype=np.int16), x, np.zeros(model.right, dtype=np.int16)])[offsets[i]:offsets[i] + model.left + model.win + model.right] for i, x in enumerate(audio_files)]  
    print(wave16.shape)
    for x in wave16:
        print(x.shape)
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
        
    
    print("extracted phn codes", vq_code)
    print('extracted spk code', spk_code)
    
    
if __name__ == '__main__':
    speaker_id_code , speaker_file, speaker_language = speaker_breakdown()
    
    model_path = test_model
    print("loading",model_path)
    sample_rate = config.sample_rate
    test_model = model_path.split("_")[-1].split(".")[0]

    model = M.Model(4, nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()

    model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    pad_left = model.pad_left()
    pad_left_encoder = model.pad_left_encoder()
    pad_left_decoder = model.pad_left_decoder()
    pad_right = model.pad_right()
    window = 16
    '''
    for u in range(6,7,4):
        for speaker_id in range(3,29):
            linguistic_unit_single_speaker(speaker_id_code=speaker_id_code , speaker_file=speaker_file, speaker_language=speaker_language, speaker_id=speaker_id,unit_size = u)
    '''

    generation_from_inference()
