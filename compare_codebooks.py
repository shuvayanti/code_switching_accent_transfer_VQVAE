import math, pickle, os, glob
import numpy as np

PHN_CODE_PATH_siwis_model = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5_lang/siwis_552024/train/"
SPK_CODE_PATH_siwis_model = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5_lang/siwis_552024/all_siwis/"
PHN_CODE_PATH_vctk_model = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5/VCTK_753011/train/"
SPK_CODE_PATH_vctk_model = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5/VCTK_753011/all_siwis/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"

speaker_file_list = pickle.load(open(SPK_MAP,'rb'))

def speaker_breakdown(SPK_CODE_PATH):
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
                
                for code,files in speaker_file.items():
                    files = [f[1].split('.')[0] for f in files]
                    if item in files:
                        #print(i,code)
                        speaker_id_code[i]=code
                    #print(i,code)
    #print(speaker_language)
    return speaker_id_code, speaker_file, speaker_language

if __name__ == "__main__":

    speaker_id_code , speaker_file, speaker_language = speaker_breakdown(SPK_CODE_PATH_siwis_model)
    print("\nspeaker codes used in the siwis model")
    print('speaker id-code map: ',speaker_id_code)
    print('speaker language: ', speaker_language)

    speaker_id_code , speaker_file, speaker_language = speaker_breakdown(SPK_CODE_PATH_vctk_model)
    print("\nspeaker codes used in the vctk model")
    print('speaker id-code map: ',speaker_id_code)
    print('speaker language: ',speaker_language)

