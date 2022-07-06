import math, pickle, os, glob
import numpy as np
import torch
import random
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
import os.path
import modules.sys5_lang as M
import librosa
import soundfile as sf
import operator
from functools import reduce
import modules.config
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

PHN_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5_lang/siwis_552024/train/"    	#"/path/to/train/phn_codes"
PHN_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_vecs_updated/sys5_lang/siwis_552024/train/"		#"/path/to/train/phn_vectors"
SPK_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5_lang/siwis_552024/all_siwis/"	#"/path/to/siwis/speaker_codes"
SPK_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_vecs_updated/sys5_lang/siwis_552024/all_siwis/"	#"/path/to/siwis/speaker_vectors"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"				#"/path/to/siwis/speaker_map"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"					#"/path/to/siwis/time_alignment"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"				#"/path/to/siwis/normalised_wavs"

num_speakers = 31
test_model = "/home/s1995633/s1995633/dissertation/codes/multilingual_VQVAE/pre-trained/sys5_siwis_lang.43.upconv_552024.pyt"	#"/path/to/multilingual_VQVAE/pretrained/model"

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
    
    
def turn_testing_multi_speaker(speaker_id_code , speaker_file, speaker_language, turns = 1):
    list_single_speaker_EN = list(speaker_language['EN']-(speaker_language['EN'].intersection(speaker_language['IT'])))
    list_single_speaker_IT = list(speaker_language['IT']-(speaker_language['EN'].intersection(speaker_language['IT'])))
    #print(list_single_speaker_EN)
    #print(list_single_speaker_IT)
    
    output_filename = ''
    speaker_code = []
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))
    #turns = random.choice(range(1,31))
    print(turns)
    #speaker_EN = speaker_id_code[random.choice(list_single_speaker_EN)]
    #file_EN = PHN_CODE_PATH + random.choice(speaker_file[speaker_EN])[1]
    #print(file_EN)
    combined_vq_code = []
    t = 1
    while t <= turns:
        speaker_EN_id = random.choice(list_single_speaker_EN)
        speaker_IT_id = random.choice(list_single_speaker_IT)
        
        speaker_EN_code = speaker_id_code[speaker_EN_id]
        speaker_IT_code = speaker_id_code[speaker_IT_id]

        #print(speaker_EN_id)
        #print(speaker_IT_id)
        
        #print('english file: ',random.choice(speaker_file_list[speaker_EN_id]))
        file_EN = random.choice(speaker_file_list[speaker_EN_id])
        file_EN_path = PHN_CODE_PATH + file_EN +'.npy'
        #print('loaded english file :', file_EN_path)
        #print('Italian file = ',random.choice(speaker_file[speaker_IT])[1])
        file_IT = random.choice(speaker_file_list[speaker_IT_id])
        file_IT_path = PHN_CODE_PATH + file_IT +'.npy'
        #print('loaded italian file :', file_IT_path)
        
        
        if os.path.exists(file_EN_path) and os.path.exists(file_IT_path):
            #print('processing english file :', file_EN_path)
            combined_vq_code = combine_phn_codes(combined_vq_code,file_EN_path)
            #print(combined_vq_code)
            #print('processing italian file :', file_IT_path)
            combined_vq_code = combine_phn_codes(combined_vq_code,file_IT_path)
            #print('combined_vq_code')
            output_filename +=file_EN + file_IT
            t+=1
    if output_filename.startswith('EN'):
        speaker_code = speaker_EN_code
    else:
        speaker_code = speaker_IT_code
    
    lang = [output_filename.split('_')[0]]
    
    print(lang)
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    if output_filename.startswith('EN'):
        output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/multi-speaker/EN/"
        os.makedirs(output_dir, exist_ok=True)

    elif output_filename.startswith('IT'):
        output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/multi-speaker/IT/"
        os.makedirs(output_dir, exist_ok=True)
    
    print(speaker_code)
    audio = voice_conversion(combined_vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_filename+'.wav', audio, sample_rate)
       
    
    

def turn_testing_single_speaker(speaker_language, speaker_file, speaker_id_code, turns =1):
        
    list_single_speaker_EN_IT = list(speaker_language['EN'].intersection(speaker_language['IT']))
    
    output_filename = 'IT_X_0_0'
    speaker_code = []
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))
    #turns = random.choice(range(1,31))
    print('\n',turns)
    
    combined_vq_code = []
    t = 1
    while t <= turns:
        speaker_EN_IT_id = random.choice(list_single_speaker_EN_IT)
        
        speaker_EN_IT_code = speaker_id_code[speaker_EN_IT_id]

        #print(speaker_EN_id)
        #print(speaker_IT_id)
        
        #print('english file: ',random.choice(speaker_file_list[speaker_EN_id]))
        file_EN =''
        file_IT =''
        

        if output_filename.split('_')[-4] == 'IT':
            #print('last file was IT')
            while not(file_EN.startswith('EN')):
                file_EN = random.choice(speaker_file_list[speaker_EN_IT_id])
            file_EN_path = PHN_CODE_PATH + file_EN +'.npy'
            if os.path.exists(file_EN_path):            
                combined_vq_code = combine_phn_codes(combined_vq_code,file_EN_path)
            #print(combined_vq_code)
            output_filename +='_'+file_EN

        elif output_filename.split('_')[-4] == 'EN': 
            #print('started with en')       
            while not(file_IT.startswith('IT')):
                file_IT = random.choice(speaker_file_list[speaker_EN_IT_id])
            file_IT_path = PHN_CODE_PATH + file_IT +'.npy'
            if os.path.exists(file_IT_path):            
                combined_vq_code = combine_phn_codes(combined_vq_code,file_IT_path)
            output_filename +='_'+file_IT  
        
        t+=1
        #print('turn')
    speaker_code = speaker_EN_IT_code
    #print(output_filename)
    output_filename = output_filename[9:]
    print(output_filename)
    #print('combined VQ code =',combined_vq_code)
    lang = [output_filename.split('_')[0]]
    
    print(lang)
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    if output_filename.startswith('EN'):
        output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/single-speaker/EN/"
        os.makedirs(output_dir, exist_ok=True)

    elif output_filename.startswith('IT'):
        output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/single-speaker/IT/"
        os.makedirs(output_dir, exist_ok=True)
    
    print(speaker_code)
    audio = voice_conversion(combined_vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_filename+'.wav', audio, sample_rate)

def turn_testing_multi_lingual(speaker_language, speaker_file, speaker_id_code, speaker_id, turns =1):
    
    output_filename = 'XX_X_0_0'
    speaker_code = []
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))
    #print()
    #turns = random.choice(range(1,31))
    #print('\n',turns)
    
    language = ['EN','IT']
    combined_vq_code = []
    t = 0
    while t <= turns:
        
        pick = output_filename.split('_')[-4]
        while pick == output_filename.split('_')[-4]:        
            pick = random.choice(language)
        
        filename = ''
        
        while not(filename.startswith(pick)):
            filename = random.choice(os.listdir(PHN_CODE_PATH))

        file_path = PHN_CODE_PATH + filename
        combined_vq_code = combine_phn_codes(combined_vq_code,file_path)
        output_filename +='_'+filename.split('.')[0]
        
        t+=1
        #print('turn')
    #speaker_id = random.choice(list(speaker_id_code.keys()))

    speaker_code = speaker_id_code[speaker_id]
    #print(output_filename)
    output_filename = 'speaker_'+str(speaker_id)+'_'+'turns_'+str(turns)+'_'+output_filename[9:]
    print(output_filename)
    #print('combined VQ code =',combined_vq_code)
    lang = [output_filename.split('_')[4]]
    
    print(lang)
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/bi-lingual/EN-IT/"
    os.makedirs(output_dir, exist_ok=True)

    audio = voice_conversion(combined_vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_filename+'.wav', audio, sample_rate)

def turn_testing_tri_lingual(speaker_language, speaker_file, speaker_id_code, speaker_id ,turns =1):
    
    #speaker_id = list(set.intersection(speaker_language['EN'],speaker_language['IT'],speaker_language['FR']))
    
    #print('list of speakers: ', speaker_id)
    #speaker_id = random.choice(speaker_id)    
    #print('chosen speaker: ',speaker_id)
    output_filename = 'XX_X_0_0'
    speaker_code = speaker_id_code[speaker_id]
    #print('speaker id: ', speaker_id, 'spekaer code: ', speaker_code)
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))
    #print()
    #turns = random.choice(range(1,31))
    #print('\n',turns)
    
    language = ['EN','FR','DE']
    combined_vq_code = []
    t = 0
    while t <= turns:
        
        pick = output_filename.split('_')[-4]
        while pick == output_filename.split('_')[-4]: 
            pick = random.choice(language)
                    
        filename = ''
        file_path =''

        while not(filename.startswith(pick)):
            filename = random.choice(os.listdir(PHN_CODE_PATH))
            print(filename)
            
        file_path = PHN_CODE_PATH + filename

        combined_vq_code = combine_phn_codes(combined_vq_code,file_path)
        output_filename +='_'+filename.split('.')[0]
        
        t+=1
        #print('turn')
    #speaker_id = random.choice(list(speaker_id_code.keys()))
    #speaker_code = speaker_id_code[speaker_id]
    #print(output_filename)
    output_filename = 'speaker_'+str(speaker_id)+'_'+output_filename[9:]
    print(output_filename)
    #print('combined VQ code =',combined_vq_code)
    lang = [output_filename.split('_')[2]]
    
    print(lang)
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    output_dir = "/home/s1995633/s1995633/dissertation/code-switch/turns/tri-lingual/FR-EN-DE/"
    os.makedirs(output_dir, exist_ok=True)

    audio = voice_conversion(combined_vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_filename+'.wav', audio, sample_rate)
       
def linguistic_unit_single_speaker(speaker_id_code , speaker_file , speaker_language, unit_size =1):
    
    print('\nsize of unit: ', unit_size)
    output_filename = ''
    speaker_code = []
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))

    list_single_speaker_EN_IT = list(speaker_language['EN'].intersection(speaker_language['IT']))
        
    speaker_EN_IT_id = random.choice(list_single_speaker_EN_IT)
        
    speaker_EN_IT_code = speaker_id_code[speaker_EN_IT_id]

    file_EN =''
    file_EN_path = PHN_CODE_PATH + file_EN +'.npy'
    
    file_IT =''
    file_IT_path = PHN_CODE_PATH + file_IT +'.npy'
    
    
    vq_code_EN = []
    vq_code_IT = []
    
    
    while not(file_EN.startswith('EN') and os.path.exists(file_EN_path)):
        file_EN = random.choice(speaker_file_list[speaker_EN_IT_id])
        file_EN_path = PHN_CODE_PATH + file_EN +'.npy'
        
    alignment_EN = ALIGN_PATH + file_EN +'.txt'
    

    vq_code_EN = combine_phn_codes(vq_code_EN,file_EN_path)
    word_list_EN, time_segment_EN = alignment_breakdown(alignment_EN)

    while not(file_IT.startswith('IT') and os.path.exists(file_IT_path)):
        file_IT = random.choice(speaker_file_list[speaker_EN_IT_id])
        file_IT_path = PHN_CODE_PATH + file_IT +'.npy'
        
    alignment_IT = ALIGN_PATH + file_IT +'.txt'

    output_file_EN = file_EN+'_'+file_IT+'_unit_size_'+str(unit_size)+'.wav'
    output_file_IT = file_IT+'_'+file_EN+'_unit_size_'+str(unit_size)+'.wav'

    print(output_file_EN)
    print(output_file_IT)

    output_dir_EN = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit/single-speaker_final/EN/"
    os.makedirs(output_dir_EN, exist_ok=True)

    output_dir_IT = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit/single-speaker_final/IT/"
    os.makedirs(output_dir_IT, exist_ok=True)

    vq_code_IT = combine_phn_codes(vq_code_IT,file_IT_path)
    word_list_IT, time_segment_IT = alignment_breakdown(alignment_IT)

    modified_vq_code_EN = vq_code_EN
    modified_vq_code_IT = vq_code_IT

    print('original EN code info: ', len(vq_code_EN))
    print('original DE code info: ', len(vq_code_IT))
    
    code_segments_EN = word_phoneCode_map(word_list_EN, time_segment_EN, vq_code_EN)
    code_segments_IT = word_phoneCode_map(word_list_IT, time_segment_IT, vq_code_IT)
    
    print('EN code segments: ',code_segments_EN)
    print('DE code segments: ',code_segments_IT)
    
    if unit_size < len(word_list_EN) and unit_size < len(word_list_IT):
        random_index_EN = random.choice(range(len(word_list_EN)-unit_size+1))
        random_index_IT = random.choice(range(len(word_list_IT)-unit_size+1))
        
        replacement_segment_EN = code_segments_EN[random_index_EN:random_index_EN+unit_size]
        replacement_segment_IT = code_segments_IT[random_index_IT:random_index_IT+unit_size]
        print('replacement segment EN:', replacement_segment_EN)
        print('replacement segment DE:', replacement_segment_IT)

        #for i in range(len(replacement_segment_EN)):
         #   segment_EN = replacement_segment_EN[i]
          #  segment_IT = replacement_segment_IT[i]

        #modified_vq_code_EN = np.concatenate([ vq_code_EN[replacement_segment_EN[0][0]: replacement_segment_EN[-1][-1] ], vq_code_IT[ replacement_segment_IT[0][0]: replacement_segment_IT[-1][-1] ]] )
        #modified_vq_code_IT = np.concatenate([ vq_code_IT[replacement_segment_IT[0][0]: replacement_segment_IT[-1][-1] ], vq_code_EN[ replacement_segment_EN[0][0]: replacement_segment_EN[-1][-1] ]] )
        #print('Segments for EN file :', code_segments_EN[ : random_index_EN], replacement_segment_IT, code_segments_EN[ random_index_EN+unit_size : ])
        #print('Segments for DE file :', code_segments_IT[ : random_index_IT], replacement_segment_EN, code_segments_IT[ random_index_IT+unit_size : ])
    else:
        return
    
        
    #English audio file
    lang = ['EN']
    
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()

    audio_EN_segment_1 = voice_conversion(vq_code_EN[: replacement_segment_EN[0][0] ], source_LANG, [speaker_EN_IT_code])
    audio_EN_segment_2 = voice_conversion(vq_code_EN[replacement_segment_EN[-1][-1]: ], source_LANG, [speaker_EN_IT_code])

    #sf.write(output_dir_EN+output_file_EN, audio_EN, sample_rate)
    
    #Italian audio file
    lang = ['IT']
    
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()

    audio_IT = voice_conversion(vq_code_IT[ replacement_segment_IT[0][0]: replacement_segment_IT[-1][-1] ], source_LANG, [speaker_EN_IT_code])

    #sf.write(output_dir_IT+output_file_IT, audio_IT, sample_rate)
    
    audio = np.concatenate( [ audio_EN_segment_1, audio_IT, audio_EN_segment_2] )
    sf.write('code-switching.wav', audio, sample_rate)

def linguistic_unit_multi_lingual(speaker_id_code , speaker_file , speaker_language, speaker_id, unit_size =1, turns = 1):
    
    #print('\nsize of unit: ', unit_size)
    output_filename = 'XX_X_0_0'
    #speaker_code = []
    #speaker_file_list = pickle.load(open(SPK_MAP,'rb'))

    language = ['EN','FR','DE','IT']
    combined_vq_code = []
    t = 0
    while t <= turns:
        
        pick = output_filename.split('_')[-4]
        while pick == output_filename.split('_')[-4]:        
            pick = random.choice(language)
        
        filename = ''
        
        while not(filename.startswith(pick)):
            filename = random.choice(os.listdir(PHN_CODE_PATH))
        #print(filename)
        file_path = PHN_CODE_PATH + filename
        alignment = ALIGN_PATH + filename.split('.')[0] +'.txt'
        
        #print(file_path)
        
        vq_code = np.load(open(file_path,'rb'))
        word_list, time_segment = alignment_breakdown(alignment)
        
        modified_vq_code = vq_code
        #output_filename +='_'+filename.split('.')[0]

        code_segments = word_phoneCode_map(word_list, time_segment, vq_code)
        
        if unit_size < len(word_list):
            random_index = random.choice(range(len(word_list)-unit_size))
                
            segments = code_segments[random_index:random_index+unit_size]
        
            modified_vq_code= vq_code [segments[0][0]: segments[-1][-1]]
            
        else:
            continue
        if len(combined_vq_code): 
            combined_vq_code = np.concatenate( (combined_vq_code, modified_vq_code), axis = 0 )
        else :
            combined_vq_code = modified_vq_code
        output_filename +='_'+filename.split('.')[0]
        t+=1
    #print(combined_vq_code)
    #speaker_id = random.choice(list(speaker_id_code.keys()))
    speaker_code = speaker_id_code[speaker_id]
    #print('speaker code = ', speaker_code)
    output_filename = 'unit_'+str(unit_size)+'_'+'turns_'+str(turns)+'_'+'spk_'+str(speaker_id)+'_'+output_filename[9:]
    print(output_filename)
    print(turns)
    lang = [output_filename.split('_')[6]]
    
    print(lang)
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()
    
    output_dir = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/multi-lingual/EN-FR-DE-IT/"
    os.makedirs(output_dir, exist_ok=True)

    audio = voice_conversion(combined_vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_filename+'.wav', audio, sample_rate)

def voice(speaker_id_code , speaker_file, speaker_language):

    output_dir = "/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/cross_lingual_new/"
    os.makedirs(output_dir, exist_ok=True)

    best_files = open('best_files_list.txt').read().splitlines()

    
    #female_speaker_id = [0, 3, 19, 21, 24, 27, 8, 12, 36]
    #male_speaker_id = [7,4, 10, 13, 23, 26]
    #best_speaker_ids =[6, 19, 21, 10, 13, 16, 23, 12, 14, 15, 22]
    best_speaker_ids = [4]
    speaker_ids = list(speaker_language['FR'].intersection(speaker_language['IT']))
    required_spk_ids = list(set(best_speaker_ids).intersection(set(speaker_ids)))
    print(required_spk_ids)
    speaker_file_list = pickle.load(open(SPK_MAP,'rb'))    

    file_1 = random.choice(best_files)
    file_1_path = PHN_CODE_PATH + file_1 + '.npy'

    file_2 = random.choice(best_files)
    file_2_path = PHN_CODE_PATH + file_2 + '.npy'

    for speaker_id in required_spk_ids:
    
        speaker_code = speaker_id_code[speaker_id]

        file_1 = ''    
        file_2 =''    
    
        while not(file_1.startswith('EN')) or not(os.path.exists(file_1_path)):
            file_1 = random.choice(best_files)
            file_1_path = PHN_CODE_PATH + file_1 + '.npy'
    
        print('file 1: ',file_1)
            
        while not(file_2.startswith('FR')) or not(os.path.exists(file_2_path)):
            file_2 = random.choice(best_files)
            file_2_path = PHN_CODE_PATH + file_2 + '.npy'
                    
        print('file 2: ',file_2)
        #file_2_path = PHN_CODE_PATH + file_2
    
        output_file_1 = 'speaker_'+str(speaker_id)+'_'+file_1.split('.')[0] + '_'+ file_2.split('.')[0]+'.wav'
        output_file_2 = 'speaker_'+str(speaker_id)+'_'+file_2.split('.')[0] + '_'+ file_1.split('.')[0]+'.wav'
    
        print(output_file_1)
        print(output_file_2)
    
        output_vq_code_1 = combine_phn_codes(file_1_path,file_2_path)
        output_vq_code_2 = combine_phn_codes(file_2_path,file_1_path)
    
        lang = [output_file_1.split('_')[2]]    
    
        source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
        source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
        source_LANG = torch.FloatTensor(source_l_onehot).cuda()   

        audio = voice_conversion(output_vq_code_1, source_LANG, [speaker_code])
    
        sf.write(output_dir+output_file_1, audio, sample_rate)

        lang = [output_file_2.split('_')[2]]    
    
        source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
        source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
        source_LANG = torch.FloatTensor(source_l_onehot).cuda()

        audio = voice_conversion(output_vq_code_2, source_LANG, [speaker_code])
    
        sf.write(output_dir+output_file_2, audio, sample_rate)  
    
    
    '''
    speaker_id = 4
    speaker_code = speaker_id_code[speaker_id]
    
    filename =''
    
    while not(filename.startswith('EN')) or not(os.path.exists(file_path)): 
        #female_speaker = random.choice(female_speaker_id)
        #print(female_speaker)
        #print(speaker_file_list[female_speaker])       
        filename = random.choice(best_files)
        file_path = PHN_CODE_PATH + filename + '.npy'
        #print('choosing files: ',file_path)
    print('file path :', file_path)
    vq_code = np.load(open(file_path, 'rb'))

    output_file = 'speaker_'+str(speaker_id)+'_'+filename+'.wav'
   
    lang = [output_file.split('_')[2]]    
    
    source_l_onehot = [(np.arange(len(langDict)) == langDict[l]).astype(np.long) for l in lang]
    source_l_onehot = torch.FloatTensor([x for x in source_l_onehot])
    source_LANG = torch.FloatTensor(source_l_onehot).cuda()   

    audio = voice_conversion(vq_code, source_LANG, [speaker_code])

    sf.write(output_dir+output_file, audio, sample_rate)
    
    '''
    
if __name__ == "__main__":
    
    speaker_id_code , speaker_file, speaker_language = speaker_breakdown()
    
    #speaker_ids = [25, 5, 7, 26, 6, 19, 21, 8, 9, 18, 10, 13, 16, 23, 11, 12, 14, 15, 22, 20, 24] #[0, 28, 1, 2, 3, 27, 4, 17, 
    #speaker_ids = range(29)
    #print(speaker_ids)
    model_path = test_model
    print("loading",model_path)
    sample_rate = config.sample_rate
    test_model = model_path.split("_")[-1].split(".")[0]

    model = M.Model(4, nspeakers=num_speakers, rnn_dims=config.rnn_dims, fc_dims=config.fc_dims, global_decoder_cond_dims=num_speakers,upsample_factors=config.upsample_factors, normalize_vq=True, noise_x=True, noise_y=True).cuda()

    model.load_state_dict(torch.load(model_path,map_location='cuda:0'))
    
    turns = 12
    '''
    for t in range(6,turns+1):
        for i in range(6):
            turn_testing_single_speaker(speaker_id_code = speaker_id_code , speaker_file = speaker_file, speaker_language = speaker_language, turns = t)
    
    turns = 12
    for t in range(6,turns+1):
        for i in range(6):
            turn_testing_multi_speaker(speaker_id_code = speaker_id_code , speaker_file = speaker_file, speaker_language = speaker_language, turns = t)

   
    for i in range(4):
        for u in range(1,7):
            linguistic_unit_single_speaker(speaker_id_code , speaker_file, speaker_language, u)
    
    turns = 12
    for t in range(12,turns+1,4):
        for speaker_id in range(6,29):
            turn_testing_multi_lingual(speaker_id_code = speaker_id_code , speaker_file = speaker_file, speaker_language = speaker_language, speaker_id = speaker_id, turns = t)
    
    for t in range(12,turns+1, 4):
        #for speaker_id in range(15):
        turn_testing_tri_lingual(speaker_id_code = speaker_id_code , speaker_file = speaker_file, speaker_language = speaker_language, speaker_id = 28, turns = t)
    
    
    
    for u in range(8,13, 2):
        for t in range(12,turns+1, 4):        
            for speaker_id in range(4,29):
                linguistic_unit_multi_lingual(speaker_id_code = speaker_id_code , speaker_file  = speaker_file , speaker_language = speaker_language, speaker_id = speaker_id, unit_size = u, turns = t)
    '''
    for speaker_id in range(4):
        linguistic_unit_multi_lingual(speaker_id_code = speaker_id_code , speaker_file  = speaker_file , speaker_language = speaker_language, speaker_id = speaker_id, unit_size = 8, turns = 12)
    
    #for i in range(5):
    #voice(speaker_id_code = speaker_id_code , speaker_file = speaker_file, speaker_language = speaker_language)

    #linguistic_unit_single_speaker(speaker_id_code , speaker_file, speaker_language, 2)
