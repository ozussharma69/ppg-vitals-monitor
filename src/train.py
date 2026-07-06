"""
train.py
Trains a CNN+LSTM model to estimate heart rate (bpm) from wrist PPG + accelerometer
windows, using the PPG-DaLiA dataset preprocessed by preprocess.py.

Architecture (matches the Deep PPG benchmark paper this dataset comes from):
- Separate 1D-CNN branches extract local features from PPG and ACC windows
- Features are pooled to a common temporal length and concatenated
- An LSTM processes the fused temporal features
- A final FC layer regresses to a single HR value (bpm)

Split (first working version, not full LOSO yet):
- Train: subjects S1-S13
- Test/val: subjects S14-S15
Once this trains cleanly, we'll wrap it in a full leave-one-subject-out loop
in evaluate.py for the final reported metric.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")
MODEL_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "ppg_hr_model.pth")

TRAIN_SUBJECTS = [f"S{i}" for i in range(1, 14)]   # S1-S13
TEST_SUBJECTS = ["S14", "S15"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPGDaLiADataset(Dataset):
    """Loads and concatenates windows across a list of subjects."""

    def __init__(self, subject_ids):
        ppg_list, acc_list, label_list = [], [], []

        for subject_id in subject_ids:
            path = os.path.join(DATA_DIR, f"{subject_id}.npz")
            data = np.load(path)
            ppg_list.append(data["ppg"])
            acc_list.append(data["acc"])
            label_list.append(data["hr_label"])

        self.ppg = np.concatenate(ppg_list, axis=0)         # (N, 512)
        self.acc = np.concatenate(acc_list, axis=0)         # (N, 256, 3)
        self.hr_label = np.concatenate(label_list, axis=0)  # (N,)

        # Per-window normalization: PPG and ACC z-scored individually.
        # This matters a lot here since raw PPG amplitude varies by subject/skin/sensor contact.
        self.ppg = (self.ppg - self.ppg.mean(axis=1, keepdims=True)) / (
            self.ppg.std(axis=1, keepdims=True) + 1e-8
        )
        acc_mean = self.acc.mean(axis=1, keepdims=True)
        acc_std = self.acc.std(axis=1, keepdims=True) + 1e-8
        self.acc = (self.acc - acc_mean) / acc_std

    def __len__(self):
        return len(self.hr_label)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppg[idx], dtype=torch.float32).unsqueeze(0)      # (1, 512)
        acc = torch.tensor(self.acc[idx], dtype=torch.float32).permute(1, 0)     # (3, 256)
        label = torch.tensor(self.hr_label[idx], dtype=torch.float32)
        return ppg, acc, label


class PPGHRNet(nn.Module):
    def __init__(self, pooled_len=32, lstm_hidden=64):
        super().__init__()

        # PPG branch: input (batch, 1, 512)
        self.ppg_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # ACC branch: input (batch, 3, 256)
        self.acc_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Pool both branches to a shared temporal length so they can be concatenated
        self.pool = nn.AdaptiveAvgPool1d(pooled_len)

        fused_channels = 64 + 32  # ppg_cnn out channels + acc_cnn out channels
        self.lstm = nn.LSTM(input_size=fused_channels, hidden_size=lstm_hidden, batch_first=True)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, ppg, acc):
        ppg_feat = self.pool(self.ppg_cnn(ppg))   # (batch, 64, pooled_len)
        acc_feat = self.pool(self.acc_cnn(acc))   # (batch, 32, pooled_len)

        fused = torch.cat([ppg_feat, acc_feat], dim=1)     # (batch, 96, pooled_len)
        fused = fused.permute(0, 2, 1)                      # (batch, pooled_len, 96) for LSTM

        lstm_out, (h_n, _) = self.lstm(fused)
        last_hidden = h_n[-1]                                # (batch, lstm_hidden)

        hr_pred = self.head(last_hidden).squeeze(-1)         # (batch,)
        return hr_pred


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for ppg, acc, label in loader:
            ppg, acc, label = ppg.to(DEVICE), acc.to(DEVICE), label.to(DEVICE)

            pred = model(ppg, acc)
            loss = criterion(pred, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(label)
            total_samples += len(label)

    return total_loss / total_samples  # mean absolute error in bpm


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading training subjects:", TRAIN_SUBJECTS)
    train_dataset = PPGDaLiADataset(TRAIN_SUBJECTS)
    print(f"  {len(train_dataset)} training windows")

    print("Loading test subjects:", TEST_SUBJECTS)
    test_dataset = PPGDaLiADataset(TEST_SUBJECTS)
    print(f"  {len(test_dataset)} test windows")

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = PPGHRNet().to(DEVICE)
    criterion = nn.L1Loss()  # MAE, matches the standard reported metric for this task
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 30
    best_test_mae = float("inf")

    print(f"\nTraining for {n_epochs} epochs...\n")
    for epoch in range(1, n_epochs + 1):
        train_mae = run_epoch(model, train_loader, criterion, optimizer)
        test_mae = run_epoch(model, test_loader, criterion)

        print(f"Epoch {epoch:2d}/{n_epochs} | Train MAE: {train_mae:.2f} bpm | Test MAE: {test_mae:.2f} bpm")

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUT)

    print(f"\nBest test MAE: {best_test_mae:.2f} bpm")
    print(f"Best model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()