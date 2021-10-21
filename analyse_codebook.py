import math, pickle, os, glob
import numpy as np
from jiwer import wer

PHN_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_codes_updated/sys5_lang/siwis_552024/train/"
PHN_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_phn_vecs_updated/sys5_lang/siwis_552024/train/"
SPK_CODE_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_codes_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_VEC_PATH = "/home/s1995633/s1995633/dissertation/codes/vq_spk_vecs_updated/sys5_lang/siwis_552024/all_siwis/"
SPK_MAP = "/home/s1995633/s1995633/dissertation/siwis_database/speaker_index_updated/index.pkl"
ALIGN_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/alignment_updated/"
DATA_PATH = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"

speaker_file_list = pickle.load(open(SPK_MAP,'rb'))

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

def speaking_rate_duration(speaker_id_code):

    speaking_rate ={}
    speaking_rate_code = {}
    longest_files = {}
    for id in speaker_id_code.keys():
        files = speaker_file_list[id]
        spk_rate = 0
        avg_duration = 0
        #arranged_files =[]
        for f in files:
            alignment_file = ALIGN_PATH + f + '.txt'
            lines = open(alignment_file).read().strip().splitlines()
            
            n_words = len(lines)
            duration = float(lines[-1].split('\t')[-1])
            avg_duration+=duration
            spk_rate+=n_words/duration
        speaking_rate[id] = (spk_rate/len(files), duration/len(files))
        try:
            speaking_rate_code[speaker_id_code[id]].append((spk_rate/len(files), duration/len(files)))
        except:
            speaking_rate_code[speaker_id_code[id]] = [(spk_rate/len(files), duration/len(files))]

    sorted_spk_rate = {k:v for k, v in sorted(speaking_rate.items(), key=lambda item: item[1][0], reverse= False)}
    print('sorted speaking rate =\n',sorted_spk_rate,'\n')

    for code,items in speaking_rate_code.items():
        n = len(items)
        avg_rate = sum([x[0] for x in items])/n
        avg_duration = sum([x[1] for x in items])/n
        speaking_rate_code[code] = (avg_rate, avg_duration)
    
    sorted_spk_rate_code = {k:v for k, v in sorted(speaking_rate_code.items(), key=lambda item: item[1][0], reverse= False)}
    print('sorted speaking rate w.r.t code =\n',sorted_spk_rate_code,'\n')
    

def find_best_files():
    original_text_path = "/home/s1995633/s1995633/dissertation/siwis_database/txt/"
    transcript_path = "/home/s1995633/s1995633/dissertation/intelligibility/voice_conversion/female-male/"
    
    files = glob.glob(transcript_path+"*")
    #print(wer(['h','e','l','l','o'],['h','e','l','o','i']))
    wer_file = open('best_files_list.txt','w')
    for f in files:
        f = f.split('/')[-1]
        print(f)
        #f = f.split('_')
        #sprint(f)
        #if not(f.startswith('FR')):
            #print(f)
            #original_file = open(original_text_path + f[11:13]+ '/' + f.split('.')[0][11:-4] + '/' + f[11:].split('.')[0]+'.txt', 'r' ,encoding = 'utf8', errors = 'ignore').read().strip()
            #original_file_letters = [char for char in original_file]
            #print(original_file_letters)
            #transcript_letters = [char for char in open(transcript_path+f, 'r' ,encoding = 'utf8').read().strip()]
            #print(transcript_letters)
            #original_words = original_file.split(' ')
            #print(original_words)
            #transcript_words = open(transcript_path+f, 'r' ,encoding = 'utf8').read().strip().split(' ')
            #print(transcript_words)            
            #error = wer(original_words, transcript_words) 
            #if error<.03:
             #   wer_file.write(f.split('.')[0]+'\n')
            #print(f.split('.')[0]+'\t'+ str(error))
        if 'EN' in f: #f.startswith('EN'):
            print('french file\n')
            original_file = open(original_text_path + f[11:13]+ '/' + f.split('.')[0][11:-4] + '/' + f[11:].split('.')[0]+'.txt', 'r' ,encoding = 'utf8', errors = 'ignore').read().strip().split(' ')
            print('original file: ',original_file)
            transcript = open(transcript_path+f).readlines()
            #print(transcript)
            matchers = ['transcript']#,'confidence']
            matching = [s.strip().split(':')[1].strip() for s in transcript if any(xs in s for xs in matchers)]
            print('matched: ',matching)
            if matching:
                transcript_letters = matching[0].split(' ')
            print(transcript_letters)
            error = wer(original_file, transcript_letters) 
            print('error:',error)
            #print('confidence: ', 1-error)
            #print(error == float(matching[1]))
            #if error<.03:
             #   wer_file.write(f.split('.')[0]+'\n')
            
    #sorted_wer = {k:v for k, v in sorted(wer_file.items(), key=lambda item: item[1])}
    #print(sorted_wer.items())

def find_minimum_unit_size():
    text_path = "/home/s1995633/s1995633/dissertation/siwis_database/txt/*/*/*"
    files = glob.glob(text_path)
    min_unit = 9999
    max_unit = 0
    sizes = np.zeros(21)
    for f in files:
        #print(f)
        txt =  open(f, 'r' ,encoding = 'utf8', errors = 'ignore').read().strip().split(' ') 
        if len(txt) < 21:  
            sizes[len(txt)]+=1
        if len(txt)>max_unit:
            max_unit = len(txt)

        if len(txt)<min_unit:
            min_unit=len(txt)

    print('\nmaximum unit size: ',max_unit)
    print('\nminimum unit size: ',min_unit)
    print('\nsize distribution: ', sizes)
if __name__ == "__main__":
    
    speaker_id_code , speaker_file, speaker_language = speaker_breakdown()
    
    speaker_code_id ={}
    for id,code in speaker_id_code.items():
        try:
            speaker_code_id[code].append(id)
        except:
            speaker_code_id[code] = [id]

    print('speaker code id mapping :',speaker_code_id, '\n')

    
    language_probability = {}
    for code in speaker_file.keys():
        languages = {'EN':0, 'FR':0, 'IT':0, 'DE':0}
        for i,file in speaker_file[code]:
            languages[file.split('_')[0]]+=1

        language_probability[code] = dict(sorted(languages.items(), key=lambda item: item[1], reverse = True))
    
    print('probability of a speaker code of a language = \n', language_probability,'\n')
    print('speaker id code mapping = \n',speaker_id_code,'\n')
    print('speaker code langauge = \n',speaker_language,'\n')
    
    
    #speaking_rate_duration(speaker_id_code)
    
    #find_minimum_unit_size()
    
    #find_best_files()
    en_fr = list(speaker_language['EN'].intersection(speaker_language['FR']))
    en_de= list(speaker_language['EN'].intersection(speaker_language['DE']))
    en_it = list(speaker_language['EN'].intersection(speaker_language['IT']))
    fr_it = list(speaker_language['FR'].intersection(speaker_language['IT']))
    fr_de = list(speaker_language['FR'].intersection(speaker_language['DE']))
    de_it = list(speaker_language['IT'].intersection(speaker_language['DE']))

    en_fr_de = list(speaker_language['EN'].intersection(speaker_language['FR']).intersection(speaker_language['DE']))
    en_fr_it = list(speaker_language['EN'].intersection(speaker_language['FR']).intersection(speaker_language['IT']))
    en_it_de = list(speaker_language['EN'].intersection(speaker_language['IT']).intersection(speaker_language['DE']))
    it_fr_de = list(speaker_language['IT'].intersection(speaker_language['FR']).intersection(speaker_language['DE']))
    
    print('EN-FR-DE = ',en_fr_de)
    print('EN-FR-IT = ',en_fr_it)
    print('EN-IT-DE = ',en_it_de)
    print('IT-FR-DE = ',it_fr_de)

    print('EN-FR = ', en_fr)
    print('EN-DE = ', en_de)
    print('EN-IT = ', en_it)
    print('DE-FR = ', fr_de)
    print('IT-FR = ', fr_it)
    print('DE-IT = ', de_it)

    print('total bilingual = ', len(en_fr)+len(en_de)+len(en_it)+len(fr_de)+len(fr_it)+len(de_it))
    print('total trilingual = ', len(en_fr_de)+len(en_fr_it)+len(en_it_de)+len(it_fr_de))
    
    
