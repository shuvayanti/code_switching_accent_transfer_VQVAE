import numpy as np
import scipy.io
import glob
import librosa
import config
from math import exp

turns_data = "/home/s1995633/s1995633/dissertation/code-switch/turns/multi-lingual/"
unit_data =  "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/multi-lingual/EN-FR-DE-IT/"
male_voice_data = "/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/female-male/"
female_voice_data = "/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/male-female/"
cross_voice_data = "/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/cross_lingual_new/"
combine_audio_data = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/combine-audio/"
combine_phn_codes_data = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/combine-phn-codes/"
inferenced_data = "/home/s1995633/s1995633/dissertation/code-switch/linguistic_unit_new/inferenced/"

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

def load_wav(filename, encode=False):
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : 
        x = encode_16bits(x)
    return x

if __name__ == '__main__':
    
    sample_rate = config.sample_rate

    wavfiles = glob.glob(cross_voice_data+'*.wav')
    #print(wavfiles)
    
    turns_snr = open('snr_combine_audio.txt','w')

    for wav in wavfiles:
        
        audio = load_wav(wav)
        
        snr = signaltonoise(audio)
        
        f = wav.split('/')[-1]
        if 'EN' in f:
            print(f)
            print(f+'\t'+str(exp(snr)))        
            turns_snr.write(f+'\t'+str(exp(snr))+'\n')
