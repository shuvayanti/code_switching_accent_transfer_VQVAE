import os
import pickle

DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"

speaker_file_list = pickle.load(open(SPK_MAP,'rb'))

def make_wav():
    wave = open('wav.scp','w')
    
    wave_files = os.listdir(DATA_PATH)
    
    for wav in wave_files:
        utt_id = wav.split('/')[-1].split('.')[0]
        spk_id = utt_id.split('_')[2]
        wave.write(spk_id+'-'+utt_id+'\t'+DATA_PATH+wav+'\n')

def make_utt2spk():
    spk = open('utt2spk','w')
    
    wave_files = os.listdir(DATA_PATH)
    
    #print(speaker_file_list)
    utt2spk ={}

    for wav in wave_files:
        utt_id = wav.split('/')[-1].split('.')[0]
        spk_id = utt_id.split('_')[2]
        try:
            utt2spk[spk_id].append(utt_id)
        except:
            utt2spk[spk_id]=[utt_id]
    utt2spk = dict(sorted(utt2spk.items(), key=lambda item: item[0]))
    for key, value in utt2spk.items():
        for utt_id in value:
            spk.write(key+'-'+utt_id+'\t'+key+'\n')

def make_segments():
    files = os.listdir(ALIGN_PATH)
    
    segments = open('segments','w')

    for f in os.listdir(ALIGN_PATH):
        info = open(ALIGN_PATH+f).read().splitlines()
        start = info[0].split('\t')[-2]
        end = info[-1].split('\t')[-1]

        utt_id = f.split('.')[0]
        spk_id = utt_id.split('_')[2]
        
        segments.write(spk_id+'-'+utt_id+'\t'+utt_id+'\t'+start+'\t'+end+'\n')


if __name__ == "__main__":
    #make_wav()

    #make_utt2spk()

    make_segments()
