#!/bin/sh

#original_wav=/home/s1995633/s1995633/dissertation/siwis_database/normalised_output_updated

#sample_wav=/home/s1995633/s1995633/dissertation/code-switch/voice_conversion/cross_lingual_new

output_dir=/home/s1995633/s1995633/dissertation/codes/f0_tracking

input_file=$1
output_file=$2
dir=$3

echo $sample_wav/${input_file}
echo $output_dir/${output_file}

reaper -i $dir/${input_file} -f $output_dir/${output_file}.f0 -p $output_dir/${output_file}.pm -a
