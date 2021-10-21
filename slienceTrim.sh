#!/bin/bash

for FILE in ../siwis_database/wav/IT/*/*.wav
do
    #echo $FILE | cut -d '/' -f 6
    #echo $FILE | cut -d '/' 6
    #FOLDER=`echo $FILE | cut -d '/' -f 5`
    mkdir -p ../siwis_database/wav_silence_trimmed/
    NEWFILE=`echo $FILE | cut -d '/' -f 6 | cut -d '.' -f 1`
    #echo $FOLDER
    echo $NEWFILE
    sox $FILE ../siwis_database/wav_silence_trimmed/$NEWFILE.wav vad -t 6 -s 0.1 -p 0.1 reverse vad -t 4 -s 0.1 -p 0.3 reverse
done

for FILE in ../siwis_database/wav/EN/*/*.wav
do
    #echo $FILE | cut -d '/' -f 6
    #echo $FILE | cut -d '/' 6
    #FOLDER=`echo $FILE | cut -d '/' -f 5`
    mkdir -p ../siwis_database/wav_silence_trimmed/
    NEWFILE=`echo $FILE | cut -d '/' -f 6 | cut -d '.' -f 1`
    #echo $FOLDER
    echo $NEWFILE
    sox $FILE ../siwis_database/wav_silence_trimmed/$NEWFILE.wav vad -t 6 -s 0.1 -p 0.1 reverse vad -t 4 -s 0.1 -p 0.3 reverse
done

for FILE in ../siwis_database/wav/DE/*/*.wav
do
    #echo $FILE | cut -d '/' -f 6
    #echo $FILE | cut -d '/' 6
    #FOLDER=`echo $FILE | cut -d '/' -f 5`
    mkdir -p ../siwis_database/wav_silence_trimmed/
    NEWFILE=`echo $FILE | cut -d '/' -f 6 | cut -d '.' -f 1`
    #echo $FOLDER
    echo $NEWFILE
    sox $FILE ../siwis_database/wav_silence_trimmed/$NEWFILE.wav vad -t 6 -s 0.1 -p 0.1 reverse vad -t 4 -s 0.1 -p 0.3 reverse
done

for FILE in ../siwis_database/wav/FR/*/*.wav
do
    #echo $FILE | cut -d '/' -f 6
    #echo $FILE | cut -d '/' 6
    #FOLDER=`echo $FILE | cut -d '/' -f 5`
    mkdir -p ../siwis_database/wav_silence_trimmed/
    NEWFILE=`echo $FILE | cut -d '/' -f 6 | cut -d '.' -f 1`
    #echo $FOLDER
    echo $NEWFILE
    sox $FILE ../siwis_database/wav_silence_trimmed/$NEWFILE.wav vad -t 6 -s 0.1 -p 0.1 reverse vad -t 4 -s 0.1 -p 0.3 reverse
done
