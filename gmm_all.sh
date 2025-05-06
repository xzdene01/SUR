#!/bin/bash

n_mfcc=(20 30 40)
n_fbank=(26 40)

for mfcc in "${n_mfcc[@]}"; do
    echo "MFCC: $mfcc"
    input_dir="data/mfcc_${mfcc}"
    output_dir="models/gmm/mfcc_${mfcc}"
    train_manifest="${input_dir}/train_manifest.csv"
    dev_manifest="${input_dir}/dev_manifest.csv"

    python3 core/GMM.py \
        --train-manifest "$train_manifest" \
        --dev-manifest "$dev_manifest" \
        --output-dir "$output_dir" -m --minmax-norm --seed 0
done

for fbank in "${n_fbank[@]}"; do
    echo "Fbank: $fbank"
    input_dir="data/fbank_${fbank}"
    output_dir="models/gmm/fbank_${fbank}"
    train_manifest="${input_dir}/train_manifest.csv"
    dev_manifest="${input_dir}/dev_manifest.csv"

    python3 core/GMM.py \
        --train-manifest "$train_manifest" \
        --dev-manifest "$dev_manifest" \
        --output-dir "$output_dir" -m --minmax-norm
done