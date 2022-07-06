# msc_slp_thesis_codes

This project tests the limits of https://github.com/rhoposit/multilingual_VQVAE.

# Setting up the conda environment
Please follow the intstructions listed [here](https://github.com/rhoposit/multilingual_VQVAE#requirements) to set up the conda environment. 

# Preprocess the Siwis dataset

1. Silence trim the audio files. A silence trimming sample [code](https://github.com/shuvayanti/msc_slp_thesis_codes/blob/main/slienceTrim.sh) is provided.
2. Normalise the audio files. You can use your own normaliser or use this [binary file](https://github.com/shuvayanti/msc_slp_thesis_codes/blob/main/sv56demo) and the [code](https://github.com/shuvayanti/msc_slp_thesis_codes/blob/main/run_sv56.py).
3. Use the provided pre-processing script: `preprocess_vqvae.py`
4. Run `make_siwis_conditions.py` to create a proper held-out set for SIWIS data.


