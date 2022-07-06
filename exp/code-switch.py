import math, pickle, os, glob
import numpy as np
import random
#import torch
#from torch.autograd import Variable
#from torch import optim
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
#from utils import *
import sys
#import utils.env as env
import argparse
import platform
import re
import utils.logger as logger
import time
import subprocess
import os.path
#import sys5_lang as M
#import librosa
#import soundfile as sf
import operator
from functools import reduce
import config
from functools import reduce
from collections import defaultdict
from operator import itemgetter
from scipy.io.wavfile import write
from pydub import AudioSegment

def load_wav(filename, encode=False):
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : 
        x = encode_16bits(x)
    return x


PHN_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes/sys5_lang/siwis_552024/train/"
PHN_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_vecs/sys5_lang/siwis_552024/train/"
SPK_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes/sys5_lang/siwis_552024/all_siwis/"
SPK_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_vecs/sys5_lang/siwis_552024/all_siwis/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"
'''

#count=0

count =0
for f in vq_phn_files:
    print(f)
    f = np.load(open(PHN_CODE_PATH+f,'rb'))
    print(len(f))
    count+=1
    if count ==10:
        break
'''
spk_codes_ind={}
vq_phn_files = os.listdir(PHN_CODE_PATH)

for i,f in enumerate(os.listdir(SPK_CODE_PATH)):
    #print(f)
    #print(SPK_CODE_PATH+f)
    
    #f = np.load(open(PHN_CODE_PATH+f,'rb'))
    s = np.load(open(SPK_CODE_PATH+f,'rb'))
    #print(s)
    #print(f)
    try:
        spk_codes_ind[s[0][0]].append(i)
    except:
        spk_codes_ind[s[0][0]]=[i]
#print(len(spk_codes_ind))
print(spk_codes_ind.keys())
#print(spk_codes_ind[65])

#print(codes[2])

#sys.exit(0)
'''
same_speaker_audio = {}
with open("/home/s1995633/s1995633/dissertation/siwis_database/speaker_index/index2.pkl", 'rb') as f:
    index = pickle.load(f)
for i, spk_list in enumerate(index):
    if not(all([not(item.startswith('EN')) for item in spk_list]) or all([not(item.startswith('IT')) for item in spk_list])):
        same_speaker_audio[i] = spk_list

print(same_speaker_audio.keys())

'''
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
#speaker_code = [[65],[248]] #4,12,13,14

#same_speaker_audio[speaker][0]
#print('1st speaker code = ', np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code][0]],'rb')))
#print('2nd speaker code =', np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code][1]],'rb')))

#combined_vq_code = np.concatenate((np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[0][0]][0]],'rb')), np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[1][0]][1]],'rb'))))

#print('combined vq code = ',len(combined_vq_code))
#print(' audio files = ', PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[0][0]][0]], PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[0][0]][1]])

#print(len(combined_vq_code))
#speaker1 = np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[0][0]][0]],'rb'))
#speaker2 = np.load(open(PHN_CODE_PATH+vq_phn_files[spk_codes_ind[speaker_code[0][0]][1]],'rb'))
#print(len(speaker1))
#print(len(speaker2))

speaker_id_code , speaker_file, speaker_language = speaker_breakdown()

print(speaker_language)

speaker_file_list = pickle.load(open(SPK_MAP,'rb'))
'''
list_single_speaker_EN = list(speaker_language['EN']-(speaker_language['EN'].intersection(speaker_language['IT'])))
list_single_speaker_IT = list(speaker_language['IT']-(speaker_language['EN'].intersection(speaker_language['IT'])))

speaker_EN_id = random.choice(list_single_speaker_EN)
speaker_IT_id = random.choice(list_single_speaker_IT)
        
speaker_EN_code = speaker_id_code[speaker_EN_id]
speaker_IT_code = speaker_id_code[speaker_IT_id]

file_EN = random.choice(speaker_file_list[speaker_EN_id])
file_EN_path = PHN_CODE_PATH + file_EN +'.npy'

file_IT = random.choice(speaker_file_list[speaker_IT_id])
file_IT_path = PHN_CODE_PATH + file_IT +'.npy'

while not(os.path.exists(file_EN_path)):
    file_EN = random.choice(speaker_file_list[speaker_EN_id])
    file_EN_path = PHN_CODE_PATH + file_EN +'.npy'

while not(os.path.exists(file_IT_path)):   
    file_IT = random.choice(speaker_file_list[speaker_IT_id])
    file_IT_path = PHN_CODE_PATH + file_IT +'.npy'

vq_code_EN = np.load(open(file_EN_path,'rb'))
vq_code_IT = np.load(open(file_IT_path,'rb'))

alignment_EN = ALIGN_PATH + file_EN +'.txt'
alignment_IT = ALIGN_PATH + file_IT +'.txt'

word_list_EN, time_segment_EN = alignment_breakdown(alignment_EN)
word_list_IT, time_segment_IT = alignment_breakdown(alignment_IT)

print('word list EN & IT: ', word_list_EN, word_list_IT)
print('time segment EN & IT: ', time_segment_EN, time_segment_IT)

#print('VQ code English :', vq_code_EN)
#print('VQ code Italian :', vq_code_IT)

output_filename = file_EN + file_IT

print(output_filename)

code_segments_EN = word_phoneCode_map(word_list_EN, time_segment_EN, vq_code_EN)
code_segments_IT = word_phoneCode_map(word_list_IT, time_segment_IT, vq_code_IT)
    
print('EN code segments: ',code_segments_EN)
print('IT code segments: ',code_segments_IT)

print('EN silence code segments: ', vq_code_EN[code_segments_EN[0][0]: code_segments_EN[0][-1]], vq_code_EN[code_segments_EN[-1][0]: code_segments_EN[-1][-1]])
#print('IT silence code segments: ', vq_code_IT[code_segments_IT[0][0]: code_segments_IT[0][-1]], vq_code_IT[code_segments_IT[-1][0]: code_segments_IT[-1][-1]])
'''

wav_file1 = random.choice(os.listdir(DATA_PATH))
speech_file1 = AudioSegment.from_wav(DATA_PATH+wav_file1)

textgrid_file1 = wav_file1.split('.')[0]
print(textgrid_file1)
word_list1 , time_segment1 = alignment_breakdown(ALIGN_PATH+textgrid_file1+'.txt')

wav_file2 = random.choice(os.listdir(DATA_PATH))
speech_file2 = AudioSegment.from_wav(DATA_PATH+wav_file2)

textgrid_file2 = wav_file2.split('.')[0]
print(textgrid_file2)
word_list2 , time_segment2 = alignment_breakdown(ALIGN_PATH+textgrid_file2+'.txt')

print(wav_file1, wav_file2)
unit_size = 2

index = random.choice(range(min(len(word_list1),len(word_list2))-unit_size))

loc1 = time_segment1[index][0] *1000
loc2 = time_segment2[index][-1] *1000

ss1 = speech_file1[:loc1]
#ss_mid = speech_file2[loc1:loc2]
ss2 = speech_file2[loc2:]

final_speech_file = ss1 + ss2

final_speech_file.export('wave_segments_3.wav', format="wav")

#for i in range(len(combined_vq_code)):
 #   if i < len(speaker1) and speaker1[i] != combined_vq_code[i]:
  #      print('speaker 1 miss match : ', i, speaker1[i], combined_vq_code[i])
   # if i >= len(speaker1) and speaker2[i-len(speaker1)] != combined_vq_code[i]:
    #    print('speaker 2 miss-match : ', i, speaker2[i], combined_vq_code[i])


#num_speakers = 31

#test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_siwis_lang.43.upconv_552024.pyt"

'''
model_path = test_model
print("loading",model_path)
sample_rate = config.sample_rate
test_model = model_path.split("_")[-1].split(".")[0]

model = M.Model(4, nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()
    
model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
'''
#pad_left = model.pad_left()
#pad_left_encoder = model.pad_left_encoder()
#pad_left_decoder = model.pad_left_decoder()
#pad_right = model.pad_right()
#window = 16
    
# create the output directory
    
#output_dir = "/home/s1995633/s1995633/dissertation/code-switch/"
#os.makedirs(output_dir, exist_ok=True)

#source_filelist = [vq_phn_files[spk_codes_ind[speaker_code[0][0]][0]].split('.')[0], vq_phn_files[spk_codes_ind[speaker_code[0][0]][1]].split('.')[0]]

#print(source_filelist)

'''
path = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output/"
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

#print('source_gt = ', source_gt)
audio_length = sum([ len(x) for x in source_gt])

#print(audio_length)

lang = ["EN"]
langDict = {"EN":0, "FR":1, "IT":2, "DE":3}
source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
source_LANG = torch.FloatTensor(source_l_onehot).cuda()

#speaker_code = [[85]]

out_EN = model.forward_audio_from_only_codes(vq_code_EN, source_LANG, [speaker_EN_code])
out_IT = model.forward_audio_from_only_codes(vq_code_IT, source_LANG, [65])

audio_EN = out_EN[0].cpu().numpy()
audio_IT = out_IT[0].cpu().numpy()

print('type of audio array: ', type(audio_EN),type(audio_IT))

audio = np.concatenate((audio_EN,audio_IT),axis = 0)

sf.write('test_switch_conversion_2.wav', audio, sample_rate)
'''


