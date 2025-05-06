import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy.special import log_softmax
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def load_scores(path):
    # Load space-delimited file: id, hard_pred, 31 log-probs
    df = pd.read_csv(path, sep=r"\s+", header=None)
    cols = ["id", "pred"] + [f"c{i+1}" for i in range(df.shape[1]-2)]
    df.columns = cols
    return df

def normalize_cols_variance(arr):
    # input shape: (n_samples, n_features)
    # output shape: (n_samples, n_features)
    # Normalize each column to have zero mean and unit variance
    arr = arr.astype(np.float32)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std[std == 0] = 1e-8  # Avoid division by zero
    arr = (arr - mean) / std
    return arr

def main(audio_file, image_file, test_dir, output_file):
    df_a = load_scores(audio_file)
    df_i = load_scores(image_file)

    # Merge on id
    df = pd.merge(df_a[["id"]], df_a, on="id").merge(df_i[["id"]], on="id")
    scores_a = df[[f"c{i+1}" for i in range(31)]].values
    scores_i = df_i.set_index("id").loc[df["id"], [f"c{i+1}" for i in range(31)]].values

    # Normalize each column
    norm_a = normalize_cols_variance(scores_a)
    norm_i = normalize_cols_variance(scores_i)

    # Fuse scores
    fused = norm_a + norm_i

    # New hard predictions
    preds = np.argmax(fused, axis=1) + 1

    # Recompute log-probs via log-softmax
    log_probs = log_softmax(fused, axis=1)

    # Prepare output DataFrame
    out_df = pd.DataFrame(
        np.hstack([df["id"].values.reshape(-1,1), preds.reshape(-1,1), log_probs]),
        columns=["id", "pred"] + [f"c{i+1}" for i in range(31)]
    )

    # Save fused scores
    out_df.to_csv(output_file, sep=" ", index=False, header=False)

    if test_dir is None:
        print("No test directory provided, skipping evaluation.", file=sys.stderr)
        return
    
    # out_df = df_a

    # Build ground truth mapping from test_dir
    truth = {}
    for speaker in os.listdir(test_dir):
        spath = os.path.join(test_dir, speaker)
        if not os.path.isdir(spath): continue
        for fname in os.listdir(spath):
            if not fname.lower().endswith(".wav"): continue
            key = os.path.splitext(fname)[0]
            truth[key] = int(speaker)

    y_true = []
    y_pred = []
    for idx, row in out_df.iterrows():
        file_id = os.path.splitext(row["id"])[0]
        if file_id in truth:
            y_true.append(truth[file_id])
            y_pred.append(int(row["pred"]))
    if len(y_true) == 0:
        print("No matching files found in test directory for evaluation.", file=sys.stderr)
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc*100:.2f}% ({len(y_true)} samples)", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse audio and image scores and evaluate.")
    parser.add_argument("--audio-file", help="Path to audio scores file")
    parser.add_argument("--image-file", help="Path to image scores file")
    parser.add_argument("--test-dir",   help="Path to test folder with speaker subdirs")
    parser.add_argument("--output",     help="Output fused scores file", default="fuse.csv")
    args = parser.parse_args()
    main(args.audio_file, args.image_file, args.test_dir, args.output)