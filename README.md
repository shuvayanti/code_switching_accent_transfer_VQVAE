# msc_slp_thesis_codes

This project tests the limits of https://github.com/rhoposit/multilingual_VQVAE.

# Setting up the conda environment
Please follow the intstructions listed [here](https://github.com/rhoposit/multilingual_VQVAE#requirements) to set up the conda environment. 

# Preprocess the Siwis dataset
1. Silence trim the audio files. A silence trimming sample code `silenceTrim.sh` is provided.
2. Normalise the audio files. You can use your own normaliser or use the binary file `sv56demo` and `run_sv56.py`.
3. Use the provided pre-processing script: `preprocess_vqvae.py`
4. Run `make_siwis_conditions.py` to create a proper held-out set for SIWIS data.

# Pretrained models
Please download the [pretrained models](https://github.com/rhoposit/multilingual_VQVAE/tree/main/pre-trained) before running the inference code. 

Run `inference_codegen.py` to obtain the speaker and language VQ-codes.

Run `code_Switching-Voice_conversion.py` to generate samples by combining langauge codes and speaker codes.

# Acknowledgements
All the codes in `modules`,`make_siwis_conditions.py`,`preprocess_vqvae.py`,`run_sv56.py` are borrowed from https://github.com/rhoposit/multilingual_VQVAE

# LICENSE
MIT License

Copyright (c) 2022 Shuvayanti Das

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
