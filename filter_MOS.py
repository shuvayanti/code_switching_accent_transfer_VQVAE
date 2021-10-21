import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    csv_file = pd.read_csv('MOS_Estimation/MOS_unit_multilingual.csv')
    print(csv_file['wav_file'].dtypes)
    #csv_file['wav_file']=csv_file['wav_file'].astype(str)
    print(csv_file['wav_file'].dtypes)
    EN_files = csv_file[csv_file['predicted_mos']>3.5]# and csv_file['wav_file'].str.contains('EN')]
    #EN_files = csv_file[csv_file['wav_file'].str.contains('EN')]
    #print(EN_files[EN_files['wav_file'].str.contains('FR')])
    #print(EN_files[ not(EN_files['wav_file'].str.contains('IT') and not(EN_files['wav_file'].str.contains('DE')) )])
    print(EN_files[EN_files['wav_file'].str.contains('turns')])
    #print(EN_files)
