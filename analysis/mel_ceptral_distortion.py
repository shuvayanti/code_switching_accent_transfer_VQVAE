import os
import math
import glob
import librosa
import pyworld
import pysptk
import numpy as np
import config
import matplotlib.pyplot as plot

#from binary_io import BinaryIOCollection

def load_wav(wav_file, sr):
    """
    Load a wav file with librosa.
    :param wav_file: path to wav file
    :param sr: sampling rate
    :return: audio time series numpy array
    """
    wav, _ = librosa.load(wav_file, sr=sr, mono=True)

    return wav


def log_spec_dB_dist(x, y):
    log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
    diff = x - y
    
    return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

def wav2mcep_numpy(wavfile, target_directory='', alpha=0.35, fft_size=512, mcep_size=34):
    # make relevant directories
    #if not os.path.exists(target_directory):
     #   os.makedirs(target_directory)

    loaded_wav = load_wav(wavfile, sr=sample_rate)

    # Use WORLD vocoder to spectral envelope
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=sample_rate,
                                   frame_period=5.0, fft_size=fft_size)

    # Extract MCEP features
    mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                           etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    fname = os.path.basename(wavfile).split('.')[0]
    np.save(os.path.join(target_directory, fname + '.npy'),
            mgc,
            allow_pickle=False)

def average_mcd(ref, synth, cost_function):
    """
    Calculate the average MCD.
    :param ref_mcep_files: list of strings, paths to MCEP target reference files
    :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
    :param cost_function: distance metric used
    :returns: average MCD, total frames processed
    """
    min_cost_tot = 0.0
    frames_tot = 0
    
    # get the trg_ref and conv_synth speaker name and sample id
    #ref_fsplit, synth_fsplit = os.path.basename(ref).split('_'), os.path.basename(synth).split('_')
    #ref_spk, ref_id = ref_fsplit[0], ref_fsplit[-1]
    #synth_spk, synth_id = synth_fsplit[0], synth_fsplit[-1]
    
    # if the speaker name is the same and sample id is the same, do MCD
    #if ref_spk == synth_spk and ref_id == synth_id:
    # load MCEP vectors
    ref_vec = np.load(ref)
    ref_frame_no = len(ref_vec)
    synth_vec = np.load(synth)

    # dynamic time warping using librosa
    min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T, metric=cost_function)
                
    min_cost_tot += np.mean(min_cost)
    frames_tot += ref_frame_no
                
    #mean_mcd = min_cost_tot / frames_tot
    
    #return mean_mcd, frames_tot
    return min_cost_tot, frames_tot

if __name__ == '__main__':

	sample_rate = config.sample_rate
	fft_size = 512
	mcep_size = 34
	sample_wav = '/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/cross_lingual_new/'
	mcep_dir_sample = '/home/s1995633/s1995633/dissertation/codes/mceps_numpy/sample/'
	os.makedirs(mcep_dir_sample, exist_ok=True)
	mcep_dir_original = '/home/s1995633/s1995633/dissertation/codes/mceps_numpy/original/'
	os.makedirs(mcep_dir_original, exist_ok=True)

	output_file = open('MCD.txt','w')
	original_wav = '/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated/'

	wav_files = os.listdir(sample_wav)

	for wav in wav_files:
		if 'EN' in wav and not('IT' in wav or 'DE' in wav or 'FR' in wav) :
			print(wav)
			wav2mcep_numpy(sample_wav+wav, target_directory= mcep_dir_sample, fft_size=fft_size, mcep_size=mcep_size)
			wav_file= wav.split('/')[-1]
			starting_index = wav_file.index('E')
			original = wav_file[starting_index: ]
			wav2mcep_numpy(original_wav+original, target_directory=mcep_dir_original, fft_size=fft_size, mcep_size=mcep_size)

	cost_function = log_spec_dB_dist
	
	for f in os.listdir(mcep_dir_sample):
		ref= f.split('/')[-1]
		starting_index = ref.index('E')
		original = f[starting_index: ]
		vc_mcd, vc_tot_frames_used = average_mcd(mcep_dir_sample+ref, mcep_dir_original+original, cost_function)
		output_file.write(ref.split('.')[0]+'\t'+str(vc_mcd) +'\t'+str(vc_tot_frames_used)+'\n')
		print(ref.split('.')[0]+'\t'+str(vc_mcd) +'\t'+str(vc_tot_frames_used))
    #print(f'StarGAN-VC MCD = {vc_mcd} dB, calculated over a total of {vc_tot_frames_used} frames')

