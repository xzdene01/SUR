import os
import csv
import argparse
import json
import time
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

def load_manifest(manifest_path):
    items = []
    with open(manifest_path, 'r') as f:
        for row in csv.DictReader(f):
            items.append((row['feat_path'], row['speaker']))
    return items

def train_ubm(all_feats, n_components, covariance_type, max_iter, seed):
    print('Training UBM (Universal Background Model)...')
    X = np.vstack(all_feats)
    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=seed
    )
    ubm.fit(X)
    return ubm

def adapt_gmms(ubm, spk_feats, relevance):
    gmms = {}
    means0 = ubm.means_
    weights0 = ubm.weights_
    for label, feats_list in tqdm(spk_feats.items(), desc='Adapting GMMs'):
        X = np.vstack(feats_list)
        # Posterior probab that i-th frame belongs to k-th Gaussian
        # - gamma[i, k] = P(k|x_i)
        gamma = ubm.predict_proba(X)

        # Total weight of data assigned to k-th Gaussian
        # - N_k = sum(gamma[i, k])
        N_k = gamma.sum(axis=0)

        # Weighted sum of feature vectors (total feature mass assigned to k-th Gaussian)
        # - F_k = sum(gamma[i, k] * x_i)
        F_k = gamma.T.dot(X)
        
        alpha = relevance
        adapted_means = (F_k + alpha * means0) / (N_k[:, None] + alpha)
        adapted_weights = (N_k + alpha * weights0) / (N_k.sum() + alpha)
        gmm = GaussianMixture(
            n_components=ubm.n_components,
            covariance_type=ubm.covariance_type,
            random_state=ubm.random_state
        )
        gmm.weights_ = adapted_weights
        gmm.means_ = adapted_means
        gmm.covariances_ = ubm.covariances_.copy()
        if ubm.covariance_type == 'diag':
            gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)
        else:
            gmm.precisions_cholesky_ = np.linalg.cholesky(
                np.linalg.inv(gmm.covariances_)
            )
        gmms[label] = gmm
    return gmms

def evaluate_metrics(dev_data, gmms, ubm, do_znorm=False, do_minmax=False):
    y_true, y_pred = [], []
    trial_scores, trial_labels = [], []
    for feat_path, tlabel in tqdm(dev_data, desc='Scoring dev data'):
        X = np.load(feat_path)
        scores = {lbl: g.score(X) - ubm.score(X) for lbl, g in gmms.items()}
        if do_znorm:
            vals = np.array(list(scores.values())); m, s = vals.mean(), vals.std() or 1e-8
            scores = {lbl: (v - m)/s for lbl, v in scores.items()}
        if do_minmax:
            vals = np.array(list(scores.values())); vmin, vmax = vals.min(), vals.max()
            scores = {lbl: (v - vmin)/(vmax - vmin + 1e-8) for lbl, v in scores.items()}
        pred_label = max(scores, key=scores.get)
        y_true.append(str(tlabel)); y_pred.append(str(pred_label))
        for lbl, sc in scores.items():
            trial_scores.append(sc)
            trial_labels.append(1 if str(lbl) == str(tlabel) else 0)
    cls_report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(gmms.keys()))
    fpr, tpr, thresholds = roc_curve(trial_labels, trial_scores)
    eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
    return cls_report, cm, eer, (fpr, tpr), (trial_scores, trial_labels)

def train(train_manifest, dev_manifest, output_dir,
          n_components, covariance_type,
          max_iter, relevance,
          z_norm, minmax_norm, seed):
    random.seed(seed); np.random.seed(seed)

    # Fit on training data
    data = load_manifest(train_manifest)
    spk_feats, all_feats = {}, []
    for path, label in tqdm(data, desc='Loading train data'):
        feats = np.load(path)
        spk_feats.setdefault(label, []).append(feats)
        all_feats.append(feats)
    ubm = train_ubm(all_feats, n_components, covariance_type, max_iter, seed)
    gmms = adapt_gmms(ubm, spk_feats, relevance)

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    # Save hyperparams
    with open(os.path.join(out_dir, 'hyperparams.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Save GMMs
    with open(os.path.join(out_dir, 'ubm.pkl'), 'wb') as f:
        pickle.dump(ubm, f)
    for lbl, g in gmms.items():
        with open(os.path.join(out_dir, f'gmm_{lbl}.pkl'), 'wb') as f:
            pickle.dump(g, f)

    # Evaluate on dev data
    dev_data = load_manifest(dev_manifest)
    report, cm, eer, (fpr, tpr), (scores, labels) = evaluate_metrics(
        dev_data, gmms, ubm, z_norm, minmax_norm
    )
    metrics = {
        'accuracy': report['accuracy'],
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'EER': eer
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot and save confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))

    # Plot and save ROC curve
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'EER={eer:.3f}')
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'))

    print(f'Results saved in {out_dir}: Acc={report["accuracy"]:.3f}, EER={eer:.3f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--dev-manifest',   required=True)
    parser.add_argument('--output-dir',     required=True)
    parser.add_argument('--n-components',   type=int,   default=64)
    parser.add_argument('--covariance-type',
                   choices=['diag','tied','full','spherical'], default='diag')
    parser.add_argument('--max-iter',       type=int,   default=200)
    parser.add_argument('--relevance',      type=float, default=16.0)
    parser.add_argument('--z-norm',         action='store_true')
    parser.add_argument('--minmax-norm',    action='store_true')
    parser.add_argument('--seed',           type=int,   default=0)
    args = parser.parse_args()

    train(
        args.train_manifest,
        args.dev_manifest,
        args.output_dir,
        args.n_components,
        args.covariance_type,
        args.max_iter,
        args.relevance,
        args.z_norm,
        args.minmax_norm,
        args.seed
    )
