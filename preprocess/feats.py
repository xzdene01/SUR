import os
import csv
import glob
import tqdm
import librosa
import argparse
import numpy as np

def extract_mfcc(y, sr,
                 n_mfcc=13,
                 n_fft=400,
                 hop_length=160):
    # y = y[2 * sr:]
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )                                               # (n_mfcc, T)
    delta = librosa.feature.delta(mfcc)             # (n_mfcc, T)
    delta2 = librosa.feature.delta(mfcc, order=2)   # (n_mfcc, T)
    feats = np.vstack([mfcc, delta, delta2])        # (3 * n_mfcc, T)
    return feats.T                                  # (T, 3 * n_mfcc)

def extract_fbank(y, sr,
                  n_fbanks=26,
                  n_fft=400,
                  hop_length=160):
    fbank = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_fbanks,
        n_fft=n_fft,
        hop_length=hop_length
    )                                                   # (n_fbanks, T)
    log_fbank = librosa.power_to_db(fbank, ref=np.max)  # (n_fbanks, T)
    delta = librosa.feature.delta(log_fbank)            # (n_fbanks, T)
    delta2 = librosa.feature.delta(log_fbank, order=2)  # (n_fbanks, T)
    feats = np.vstack([log_fbank, delta, delta2])       # (3 * n_fbanks, T)
    return feats.T                                      # (T, 3 * n_fbanks)

def extract_fft(y, sr, n_fft=512):
    """
    Extract FFT features from a single frame of audio.
    This feature extraction method is used for 1D CNN training. If tehre is no error in the pipeline the input should
    already be a single frame, but we make sure to fix the length to be exactly 1 frame.
    """
    y = librosa.util.fix_length(y, size=sr)             # (T,)
    stft = librosa.stft(y, n_fft=n_fft,
                        hop_length=n_fft,
                        win_length=n_fft,
                        center=False,
    )                                                   # (n_fft//2+1, 1)

    # take the magnitude of the STFT (complex -> real)
    mag = np.abs(stft)                                  # (n_fft//2+1, 1)
    mag = mag[:, 0]                                     # (n_fft//2+1,)

    # add back time dimension for expected CNN input shape: (T, D)
    feats = mag[np.newaxis, :]                          # (1, n_fft//2+1)
    return feats                                        # (1, n_fft//2+1)

def apply_cmvn(feats):
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (feats - mean) / std

def process_file(wav_path, input_root, output_root, n_mfcc=None, n_fbanks=None, n_fft=None):
    speaker = os.path.basename(os.path.dirname(wav_path))
    fname = os.path.splitext(os.path.basename(wav_path))[0]
    utt_id = fname

    y, sr = librosa.load(wav_path, sr=16000)
    if n_mfcc:
        feats = extract_mfcc(y, sr, n_mfcc=n_mfcc)
    elif n_fbanks:
        feats = extract_fbank(y, sr, n_fbanks=n_fbanks)
    elif n_fft:
        feats = extract_fft(y, sr, n_fft=n_fft)
    else:
        raise ValueError("Either n_mfcc, n_fbanks, n_mels, or n_fft must be specified.")
    
    # When using FFT features, do not apply normalization - we have only one frame
    # this will be done after in the actual CNN
    if not n_fft:
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

def process_folder(input_dir, output_dir, n_mfcc=None, n_fbanks=None, n_fft=None):
    os.makedirs(output_dir, exist_ok=True)
    # for subset in [!train", "dev"]:
    for subset in ["eval"]:
        in_dir = os.path.join(input_dir, subset)
        if not os.path.isdir(in_dir):
            print(f"Input directory {in_dir} does not exist.")
            continue

        wav_paths = glob.glob(os.path.join(in_dir, '**', '*.wav'), recursive=True)

        manifest = []
        for wav_path in tqdm.tqdm(wav_paths, desc=f"Processing {subset}"):
            manifest.append(process_file(wav_path, input_dir, output_dir, n_mfcc, n_fbanks, n_fft))
        
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
    parser = argparse.ArgumentParser(description="Feature extraction for speech data")
    parser.add_argument("-i", "--input", required=True, help="Input root directory containing WAV files")
    parser.add_argument("-o", "--output", required=True, help="Output directory for features and manifest")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n-mfcc",      type=int, help="Number of MFCC")
    group.add_argument("--n-fbanks",    type=int, help="Number of filterbank bins")
    group.add_argument("--n-fft",       type=int, help="Number of FFT points")
    args = parser.parse_args()

    process_folder(
        args.input, args.output,
        n_mfcc=args.n_mfcc,
        n_fbanks=args.n_fbanks,
        n_fft=args.n_fft
    )

if __name__ == '__main__':
    main()