import os
import json
import time
import random
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from utils import load_manifest
from eval_gmm import evaluate_metrics, save_results

def train_ubm(all_feats, n_components, covariance_type, max_iter, seed):
    print("Training UBM (Universal Background Model)...")
    X = np.vstack(all_feats)
    ubm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=seed,
        verbose=1,
        verbose_interval=10,
    )
    ubm.fit(X)
    return ubm

def adapt_gmms(ubm, spk_feats, relevance, do_weights=False, do_means=False, do_covar=False):
    gmms = {}
    means0 = ubm.means_
    weights0 = ubm.weights_
    covars0 = ubm.covariances_

    for label, feats_list in tqdm(spk_feats.items(), desc="Adapting GMMs"):
        X = np.vstack(feats_list)

        prior = ubm.predict_proba(X)
        n_i = prior.sum(axis=0)
        E_i = prior.T.dot(X) / n_i[:, None]
        E_i2 = prior.T.dot(X**2) / n_i[:, None]
        alpha_i = n_i / (n_i + relevance)

        adapted_weights = (alpha_i * (n_i / X.shape[0])) + ((1 - alpha_i) * weights0)
        adapted_weights /= adapted_weights.sum()
        adapted_means = (alpha_i[:, None] * E_i) + ((1 - alpha_i[:, None]) * means0)
        adapted_covariances = (alpha_i[:, None] * E_i2) + ((1 - alpha_i[:, None]) * (ubm.covariances_ ** 1 + means0 ** 2)) - (adapted_means ** 2)

        gmm = GaussianMixture(
            n_components=ubm.n_components,
            covariance_type=ubm.covariance_type,
            random_state=ubm.random_state
        )
        gmm.weights_ = adapted_weights if do_weights else weights0
        gmm.means_ = adapted_means if do_means else means0
        gmm.covariances_ = adapted_covariances if do_covar else covars0

        # Set the precisions based on covariance type after manual adaptation of parameters
        if ubm.covariance_type == "diag":
            gmm.precisions_cholesky_ = 1.0 / np.sqrt(gmm.covariances_)
        else:
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
        gmms[label] = gmm
    return gmms

def train(train_manifest, dev_manifest, output_dir, n_components, covariance_type, max_iter, relevance, minmax_norm,
          seed, weights, means, covar):
    if seed == 0:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)

    # Fit on training data
    data = load_manifest(train_manifest)
    spk_feats, all_feats = {}, []
    for path, label in tqdm(data, desc="Loading train data"):
        feats = np.load(path)
        spk_feats.setdefault(label, []).append(feats)
        all_feats.append(feats)
    ubm = train_ubm(all_feats, n_components, covariance_type, max_iter, seed)
    gmms = adapt_gmms(ubm, spk_feats, relevance, weights, means, covar)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(output_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    # Save hyperparams
    with open(os.path.join(out_dir, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Save GMMs
    with open(os.path.join(out_dir, "ubm.pkl"), "wb") as f:
        pickle.dump(ubm, f)
    for lbl, g in gmms.items():
        with open(os.path.join(out_dir, f"gmm_{lbl}.pkl"), "wb") as f:
            pickle.dump(g, f)

    # Evaluate and save results on dev data
    dev_data = load_manifest(dev_manifest)
    report, cm, eer, roc, det = evaluate_metrics(dev_data, gmms, ubm, do_minmax=minmax_norm)
    save_results(report, cm, eer, roc, det, out_dir)
    print(f"Results saved in {out_dir}: Acc={report["accuracy"]:.3f}, EER={eer:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GMM-UBM for SID")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest",   required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--n-components",   type=int,   default=256)
    parser.add_argument("--covariance-type",
                   choices=["diag","tied","full","spherical"], default="diag")
    parser.add_argument("--max-iter",       type=int,   default=300)
    parser.add_argument("--relevance",      type=float, default=16.0)
    parser.add_argument("--minmax-norm",    action="store_true")
    parser.add_argument("--seed",           type=int,   default=42, help="Set seed for reproducibility (0 for random)")
    parser.add_argument("-w", "--weights",  action="store_true")
    parser.add_argument("-m", "--means",    action="store_true")
    parser.add_argument("-c", "--covar",    action="store_true")
    args = parser.parse_args()

    train(
        args.train_manifest,
        args.dev_manifest,
        args.output_dir,
        args.n_components,
        args.covariance_type,
        args.max_iter,
        args.relevance,
        args.minmax_norm,
        args.seed,
        args.weights,
        args.means,
        args.covar,
    )
