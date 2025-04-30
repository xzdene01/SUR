import os
import json
import pandas as pd

root_dir = "./models/gmm"

results = pd.DataFrame()
for folder in os.listdir(root_dir):
    print(f"Processing trials in: {folder}")

    for subfolder in os.listdir(os.path.join(root_dir, folder)):
        with open(os.path.join(root_dir, folder, subfolder, "metrics.json")) as f:
            metrics = json.load(f)

        row = {
            "run_id": subfolder,
            "feats": folder,
            "accuracy": metrics["accuracy"],
            "EER": metrics["EER"],
        }

        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)

results.to_csv("results.csv", index=False)