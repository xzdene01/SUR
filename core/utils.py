import csv

def load_manifest(manifest_path):
    items = []
    with open(manifest_path, "r") as f:
        for row in csv.DictReader(f):
            items.append((row["feat_path"], (row["speaker"])))
    return items