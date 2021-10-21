from scipy.io import wavfile
import glob, os, sys
import multiprocessing as mp

def process(f):
    fname1 = f.split("/")[-1]
    fname = ".".join(fname1.split(".")[:-1])
    #print('fname=',fname)
    infile = wav_dir+"/"+fname+".wav"
    
    sox1_fname = sox1_dir+"/"+fname+".raw"
    sox1_command = "sox "+f+" -b 16 -r 16000 -t raw "+sox1_fname
    os.system(sox1_command)
    
    sv56_fname = sv56_dir+"/"+fname+"_sv56.raw"
    sv56_path = "/home/s1995633/s1995633/dissertation/codes/sv56demo"
    sv56_command = sv56_path+" -log loggy.log -q -lev -26 -sf 16000 "+sox1_fname+" "+sv56_fname+" 640"
    os.system(sv56_command)
    sox2_fname = outdir+"/"+fname+".wav"
    sox2_command = "sox -r 16000 -t raw -e signed -b 16 -c 1 "+sv56_fname+" "+sox2_fname
    os.system(sox2_command)
    #os._exit(os.EX_OK)
    #sys.exit()
    #f.close()
   
#for dirs in os.listdir("/home/s1995633/s1995633/dissertation/siwis_database/wav_silence_trimmed/"):
 #   print(dirs)
outdir = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/"   #sys.argv[4]
sv56_dir = "/home/s1995633/s1995633/dissertation/codes/sv56_updated/"   #sys.argv[3]
sox1_dir ="/home/s1995633/s1995633/dissertation/siwis_database/sox_output_updated/"      #sys.argv[2]
wav_dir = "/home/s1995633/s1995633/dissertation/siwis_database/wav_silence_trimmed"        #sys.argv[1]

os.makedirs(sv56_dir, exist_ok=True)
os.makedirs(outdir, exist_ok=True)
os.makedirs(sox1_dir, exist_ok=True)

files = glob.glob(wav_dir+"/*.wav")
pool = mp.Pool(100)
pool.map(process, files)
pool.terminate()
    #break
'''
for dirs in os.listdir("/home/s1995633/s1995633/dissertation/siwis_database/wav1/IT"):
    print(dirs)
    outdir = "/home/s1995633/s1995633/dissertation/siwis_database/normalised_output/"      #sys.argv[4]
    sv56_dir = "/home/s1995633/s1995633/dissertation/codes/sv56/"     #sys.argv[3]
    sox1_dir ="/home/s1995633/s1995633/dissertation/siwis_database/sox_output/"      #sys.argv[2]
    wav_dir = "/home/s1995633/s1995633/dissertation/siwis_database/wav1/IT/"+dirs           #sys.argv[1]

    os.makedirs(sv56_dir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(sox1_dir, exist_ok=True)

    files = glob.glob(wav_dir+"/*.wav")
    pool = mp.Pool(100)
    pool.map(process, files)
    pool.terminate()
'''

    
