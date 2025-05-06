import os, json, time, argparse
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.special import logsumexp
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, det_curve

from utils import load_manifest

results = pd.DataFrame()

def load_models(model_dir):
    # Load UBM
    ubm_path = os.path.join(model_dir, "ubm.pkl")
    ubm = pickle.load(open(ubm_path, "rb"))

    # Load GMMs
    gmms = {}
    for fname in sorted(os.listdir(model_dir)):
        if fname.startswith("gmm_") and fname.endswith(".pkl"):
            label = fname[len("gmm_"):-4]
            gmms[label] = pickle.load(open(os.path.join(model_dir, fname), "rb"))
    return ubm, gmms

def evaluate_metrics(data_list, gmms, ubm, do_minmax=False, create_csv=False):
    global results
    y_true, y_pred = [], []
    trial_scores, trial_labels = [], []

    for feat_path, true_label in tqdm(data_list, desc="Scoring test data"):
        X = np.load(feat_path)

        scores = {lbl: g.score(X) - ubm.score(X) for lbl, g in gmms.items()}
        if do_minmax:
            vals = np.array(list(scores.values())); vmin, vmax = vals.min(), vals.max()
            scores = {lbl: (v - vmin)/(vmax - vmin + 1e-8) for lbl, v in scores.items()}
        pred = max(scores, key=scores.get)

        if create_csv:
            score_list = [scores[str(lbl)] for lbl in range(1, 32)]
            Z = logsumexp(np.array(score_list))
            log_probabs = score_list - Z
            row = [os.path.basename(feat_path)[:-4], pred] + log_probabs.tolist()
            results = pd.concat([results, pd.DataFrame([row], columns=["file", "pred"] + [i for i in range(1, 32)])], ignore_index=True)

        y_true.append(str(true_label))
        y_pred.append(str(pred))
        for lbl, sc in scores.items():
            trial_scores.append(sc)
            trial_labels.append(1 if str(lbl) == str(true_label) else 0)

    report = classification_report(y_true, y_pred, output_dict=True)
    labels = list(gmms.keys())
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fpr, tpr, _ = roc_curve(trial_labels, trial_scores)
        det_fpr, det_fnr, _ = det_curve(trial_labels, trial_scores)
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
        return report, cm, eer, (fpr, tpr), (det_fpr, det_fnr)
    except ValueError as e:
        print(f"Error in evaluation: {e}")
        return report, None, None, None, None

def save_results(report, cm, eer, roc, det, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # metrics JSON
    metrics = {
        "accuracy": report.get("accuracy"),
        "EER": eer,
        "classification_report": report
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png")); plt.close(fig)

    # ROC curve
    fpr, tpr = roc
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"EER={eer:.3f}")
    ax.plot([0,1], [0,1], "--", color="gray")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))

    # DET curve
    det_fpr, det_fnr = det
    fig, ax = plt.subplots()
    ax.plot(det_fpr, det_fnr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("False Negative Rate")
    plt.savefig(os.path.join(output_dir, "det_curve.png"))

def run_evaluation(model_dir, test_manifest, output_root, do_minmax=False, create_csv=False):
    global results
    ubm, gmms = load_models(model_dir)
    data = load_manifest(test_manifest)

    report, cm, eer, roc, det = evaluate_metrics(data, gmms, ubm, do_minmax, create_csv)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_root, ts)
    try:
        save_results(report, cm, eer, roc, det, out_dir)
        print(f"Results saved in {out_dir}: Acc={report['accuracy']:.3f}, EER={eer:.3f}")
    except Exception as e:
        print(f"Error saving results: {e}")

    if create_csv:
        results.to_csv(os.path.join(create_csv), index=False, sep=" ", header=False)
        print(f"Results saved in audio_gmm_ubm.csv")

    return report, cm, eer, roc, det

def main():
    parser = argparse.ArgumentParser(description="Evaluate GMM model on SID test set")
    parser.add_argument("--model-dir",      required=True)
    parser.add_argument("--test-manifest",  required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--minmax-norm",    action="store_true")
    parser.add_argument("--create-csv",     type=str, default=None)
    args = parser.parse_args()

    run_evaluation(
        args.model_dir,
        args.test_manifest,
        args.output_dir,
        do_minmax=args.minmax_norm,
        create_csv=args.create_csv
    )

if __name__ == '__main__':
    main()
