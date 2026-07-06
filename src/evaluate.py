"""
evaluate.py
Full leave-one-subject-out (LOSO) cross-validation for the PPG HR estimation model.

For each of the 15 subjects:
  - Train a fresh model on the other 14 subjects
  - Test on the held-out subject
  - Record overall MAE and per-activity MAE

This is the standard evaluation protocol for PPG-DaLiA (matches the Deep PPG
paper), and gives an honest estimate of how the model generalizes to a
completely unseen person -- much more meaningful than a single fixed split.

Note: this trains 15 separate models, so it takes ~15x longer than train.py.
We use fewer epochs per fold (15 instead of 30) since train.py showed test MAE
plateauing well before epoch 30.

Outputs:
  reports/loso_results.csv       - per-fold overall MAE
  reports/loso_activity_mae.csv  - per-fold, per-activity MAE
  Printed summary: mean +/- std MAE across all 15 folds
"""

import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse the dataset and model defined in train.py rather than duplicating them
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import PPGDaLiADataset, PPGHRNet, DEVICE

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
ALL_SUBJECTS = [f"S{i}" for i in range(1, 16)]
N_EPOCHS_PER_FOLD = 15  # fewer than train.py's 30, since MAE plateaus earlier


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

    return total_loss / total_samples


def evaluate_per_activity(model, test_dataset, criterion):
    """Returns a dict of {activity_id: mae} for the held-out subject."""
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for ppg, acc, label in loader:
            ppg, acc = ppg.to(DEVICE), acc.to(DEVICE)
            pred = model(ppg, acc).cpu().numpy()
            all_preds.append(pred)
            all_labels.append(label.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    abs_errors = np.abs(all_preds - all_labels)

    activity_mae = {}
    for activity_id in np.unique(test_dataset.activity_raw):
        mask = test_dataset.activity_raw == activity_id
        if mask.sum() > 0:
            activity_mae[int(activity_id)] = float(abs_errors[mask].mean())

    return activity_mae


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    fold_results = []
    activity_results = []  # list of dicts, one per fold

    for held_out in ALL_SUBJECTS:
        train_subjects = [s for s in ALL_SUBJECTS if s != held_out]

        print(f"=== Fold: held-out subject {held_out} ===")
        train_dataset = PPGDaLiADataset(train_subjects)
        test_dataset = PPGDaLiADataset([held_out])

        # Attach raw (unwindowed-index) activity array for per-activity breakdown
        test_dataset.activity_raw = np.load(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", f"{held_out}.npz")
        )["activity"]

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model = PPGHRNet().to(DEVICE)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, N_EPOCHS_PER_FOLD + 1):
            train_mae = run_epoch(model, train_loader, criterion, optimizer)

        test_mae = run_epoch(model, test_loader, criterion)
        print(f"  Final train MAE: {train_mae:.2f} bpm | Held-out ({held_out}) test MAE: {test_mae:.2f} bpm\n")

        fold_results.append({"subject": held_out, "test_mae": test_mae})
        activity_mae = evaluate_per_activity(model, test_dataset, criterion)
        activity_results.append({"subject": held_out, **activity_mae})

    # --- Save overall per-fold results ---
    overall_path = os.path.join(REPORTS_DIR, "loso_results.csv")
    with open(overall_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "test_mae"])
        writer.writeheader()
        writer.writerows(fold_results)

    # --- Save per-activity results ---
    all_activity_ids = sorted({k for row in activity_results for k in row if k != "subject"})
    activity_path = os.path.join(REPORTS_DIR, "loso_activity_mae.csv")
    with open(activity_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject"] + [f"activity_{a}" for a in all_activity_ids])
        writer.writeheader()
        for row in activity_results:
            writer.writerow({"subject": row["subject"], **{f"activity_{k}": v for k, v in row.items() if k != "subject"}})

    # --- Print summary ---
    all_mae = [r["test_mae"] for r in fold_results]
    print("=" * 50)
    print(f"LOSO Cross-Validation Complete ({len(ALL_SUBJECTS)} folds)")
    print(f"Mean MAE: {np.mean(all_mae):.2f} bpm  |  Std: {np.std(all_mae):.2f} bpm")
    print(f"Best fold: {min(fold_results, key=lambda r: r['test_mae'])}")
    print(f"Worst fold: {max(fold_results, key=lambda r: r['test_mae'])}")
    print(f"\nResults saved to:\n  {overall_path}\n  {activity_path}")


if __name__ == "__main__":
    main()