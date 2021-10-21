import matplotlib.pyplot as plt
from collections import defaultdict
import math
def process_turns(file_lines):
    
    data = defaultdict(dict)
    for line in file_lines:
        line = line.strip()
        #print(line)
        vals = line.split('\t')
        #print(vals)
        data[int(vals[1])][int(vals[0])]=float(vals[2])
    #print(data)
    
    avg_turns = {}
    speaker_id_trend ={}

    for key,value in data.items():
        #print(key,value)
        snr = list(value.values())
        #print(snr)
        avg_turns[key] = sum(snr)/len(snr)
        
        for spk_id,val in value.items():
            if key == 4:
                speaker_id_trend[spk_id]=[val]
            elif key in [8,12]:
                speaker_id_trend[spk_id].append(val)
            else:
                continue
    
    print('\naverage turns :',avg_turns)
    speaker_id_trend = {k: v for k, v in sorted(speaker_id_trend.items(), key=lambda item: item[0])}
    print('\nspeaker id trend: ',speaker_id_trend)
    
    for key,value in speaker_id_trend.items():
        plt.bar(range(len(value)), value, color = ['r','g','b'])

    plt.savefig('turns_trend.png')


if __name__ =='__main__':

    snr_file = open('snr_turns.txt').readlines()
    #print(snr_file)

    process_turns(snr_file)
