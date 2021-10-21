## ecooper 2019-06-27
## try the Watson ASR API

import os, sys, glob

import json

## 1. run the API
# my api_key is P0ohlQATWCschQsC2ClMmdy1kvFJuIWC3HgHa6s7zOE5
# my url is https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/17b93366-e2ec-4638-bbff-135caa86d68a

#wavpath = "jasons_audio/"
#dest = "jasons_transcripts_watson/"
#orig = "jasons_orig_txt/"

wavpath = "/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/female-male/"

dest = "/home/s1995633/s1995633/dissertation/intelligibility/voice_conversion/female-male/"
os.makedirs(dest, exist_ok=True)

model_name = {'EN':'en-US_BroadbandModel', 'FR':'fr-FR_BroadbandModel'}
wavs = glob.glob(wavpath+"*.wav")

for wav in wavs:
    #print(wav)
    if 'EN' in wav:# and not('IT' in wav or 'DE' in wav):
        lang = wav.split('/')[-1].split('_')[2]
            
        # mv wav to temp name
        print(wav)
        cmd = "cp "+wav+" temp.wav"
        print(cmd)
        os.system(cmd)
        if lang == 'EN':
            cmd = 'curl -X POST -u "apikey:_csfGLlfalPwNdmyoWUm-fpflpYfn0mf0wZ9zYMyf1by" --header "Content-Type: audio/wav" --data-binary @' + "temp.wav" + ' "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/17b93366-e2ec-4638-bbff-135caa86d68a/v1/recognize?model=en-US_NarrowbandModel"'
        elif lang == 'FR':
            cmd = 'curl -X POST -u "apikey:_csfGLlfalPwNdmyoWUm-fpflpYfn0mf0wZ9zYMyf1by" --header "Content-Type: audio/wav" --data-binary @' + "temp.wav" + ' "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/17b93366-e2ec-4638-bbff-135caa86d68a/v1/recognize?model=fr-FR_BroadbandModel"'

        cmd += ' > ' + dest + wav.split("/")[-1] + '.transcript'
        print(cmd)
        os.system(cmd)
    
