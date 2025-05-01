# Heavy inspiration taken from:
#   Towards Speaker Identification with Minimal Dataset and Constrained Resources using 1D-Convolution Neural Network
#   (https://arxiv.org/pdf/2411.15082)

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import load_manifest

class FeatureDataset(Dataset):
    """
    Loads features from a manifest CSV with columns: utt_id, speaker, feat_path, num_frames
    Assumes features are stored as .npy arrays of shape (T, D).
    """
    def __init__(self, manifest_path):
        raw_entries = load_manifest(manifest_path) # List[(feat_path, speaker_str)]
        # convert 1..31 to 0..30
        self.entries = [(feat_path, int(label) - 1) for feat_path, label in raw_entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        feat_path, label = self.entries[idx]
        feats = np.load(feat_path)
        feats = torch.from_numpy(feats).float()
        return feats, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.match = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.match = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut
        identity = self.match(identity)

        # Combine
        out = out + identity
        out = self.relu(out)
        out = self.pool(out)
        return out

class SpeakerResNet(nn.Module):
    def __init__(self, input_length, num_classes=31):
        super().__init__()
        # Input channel = 1
        self.layer1 = ResidualBlock(1, 16)
        self.layer2 = ResidualBlock(16, 32)
        self.layer3 = ResidualBlock(32, 32)
        self.layer4 = ResidualBlock(32, 64)

        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=3)

        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1,1,input_length)
            out = self.layer4(self.layer3(self.layer2(self.layer1(dummy))))
            out = self.avgpool(out)
            flat_dim = out.numel()

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 1, L)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

def train_loop(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * feats.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def eval_loop(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and loaders
    train_ds = FeatureDataset(args.train_manifest)
    dev_ds = FeatureDataset(args.dev_manifest)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    # Model and training setup
    sample_feats, _ = train_ds[0]
    input_dim = sample_feats.shape[-1]
    model = SpeakerResNet(input_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # Training
    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc = train_loop(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_acc = eval_loop(model, dev_loader, criterion, device)
        scheduler.step(dev_loss)

        print(f"Epoch {epoch:02d}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
              f"Dev loss {dev_loss:.4f}, acc {dev_acc:.4f} | lr: {optimizer.param_groups[0]["lr"]:.0e}")
        if dev_acc > best_acc:
            best_acc = dev_acc
            if args.save_path is not None:
                torch.save(model.state_dict(), args.save_path)
    print(f"Best dev acc: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D-CNN SID Training")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest",   required=True)

    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--save-path",      type=str,   default=None)
    args = parser.parse_args()

    main(args)