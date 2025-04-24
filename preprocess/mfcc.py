import os
import argparse
import glob
import numpy as np
import librosa
import csv
import tqdm

def extract_mfcc(y, sr,
                 n_mfcc=13,
                 n_fft=400,
                 hop_length=160):
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, delta, delta2])
    return feats.T

def apply_cmvn(feats):
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (feats - mean) / std

def process_file(wav_path, input_root, output_root):
    speaker = os.path.basename(os.path.dirname(wav_path))
    fname = os.path.splitext(os.path.basename(wav_path))[0]
    utt_id = f"{speaker}_{fname}"

    y, sr = librosa.load(wav_path, sr=16000)
    feats = extract_mfcc(y, sr)
    feats = apply_cmvn(feats)

    rel_dir = os.path.relpath(os.path.dirname(wav_path), input_root)
    out_dir = os.path.join(output_root, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    feat_file = os.path.join(out_dir, f"{utt_id}.npy")
    np.save(feat_file, feats)

    return {
        'utt_id': utt_id,
        'speaker': speaker,
        'feat_path': feat_file,
        'num_frames': feats.shape[0]
    }

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'dev']:
        in_dir = os.path.join(input_dir, subset)
        if not os.path.isdir(in_dir):
            print(f"Input directory {in_dir} does not exist.")
            continue

        wav_paths = glob.glob(os.path.join(in_dir, '**', '*.wav'), recursive=True)

        manifest = []
        for wav_path in tqdm.tqdm(wav_paths, desc=f"Processing {subset}"):
            manifest.append(process_file(wav_path, input_dir, output_dir))
        
        manifest_file = os.path.join(output_dir, f"{subset}_manifest.csv")
        with open(manifest_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["utt_id", "speaker", "feat_path", "num_frames"]
            )
            writer.writeheader()
            for row in manifest:
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Extract MFCC features + CMVN")
    parser.add_argument("-i", "--input", required=True, help="Input root directory containing WAV files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for features and manifest")
    args = parser.parse_args()
    
    process_folder(args.input, args.output)

if __name__ == '__main__':
    main()
