#!/bin/bash

n_mfcc=(20 30 40)
n_fbanks=(26 40)

# n_ffts=(16000) # This is only for CNN use

input_dir="data/augmented"

for mfcc in "${n_mfcc[@]}"; do
    echo "MFCC: $mfcc"
    output_dir="data/mfcc_${mfcc}"
    python3 preprocess/feats.py -i ${input_dir} -o ${output_dir} --n-mfcc "$mfcc"
done

for fbank in "${n_fbanks[@]}"; do
    echo "Fbank: $fbank"
    output_dir="data/fbank_${fbank}"
    python3 preprocess/feats.py -i ${input_dir} -o ${output_dir} --n-fbanks "$fbank"
done

for fft in "${n_ffts[@]}"; do
    echo "FFT: $fft"
    output_dir="data/fft_${fft}"
    python3 preprocess/feats.py -i ${input_dir} -o ${output_dir} --n-fft "$fft"
done