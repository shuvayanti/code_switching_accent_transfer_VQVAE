import os
import subprocess
import math
import matplotlib.pyplot as plt

output_dir = '/home/s1995633/s1995633/dissertation/codes/f0_tracking/'
os.makedirs(output_dir, exist_ok=True)
output_file = open(output_dir+'RMSE.txt','w')
original_wav = '/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/'

sample_wav = '/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/cross_lingual_new/'

wav_files = os.listdir(sample_wav)

for wav in wav_files:
    if 'EN' in wav and not('IT' in wav or 'DE' in wav or 'FR' in wav) :
        print(wav)
        wav_file= wav.split('/')[-1]
        starting_index = wav_file.index('E')
        original = wav_file[starting_index: ]
        wav_name = wav_file.split('.')[0]
        print(original)
        original_name = original.split('.')[0]
        subprocess.run(["./reaper_command.sh",wav_file,wav_name,sample_wav])
        subprocess.run(["./reaper_command.sh",original,original_name,original_wav])
        sample_f0 = open(output_dir+wav_name+'.f0').read().splitlines()[7:]
        original_f0 = open(output_dir+original_name+'.f0').read().splitlines()[7:]
        
        RMSE =[]
        for i in range(min(len(sample_f0), len(original_f0))):
            sample_wav_f0 = float(sample_f0[i].split(' ')[-1])
            original_wav_f0 = float(original_f0[i].split(' ')[-1])
            
            error = (sample_wav_f0 - original_wav_f0)**2
            RMSE.append(error)

        #print(RMSE)
        print('total error= ', math.sqrt(sum(RMSE)/len(RMSE)))
        output_file.write(wav +'\t'+ str(math.sqrt(sum(RMSE)/len(RMSE)))+'\n')
        #plt.plot(RMSE)
        #plt.show()
