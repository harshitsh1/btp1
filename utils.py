"""
utils.py
========
WHY
---
Small pieces of shared plumbing (reproducibility seeding, the balanced-
accuracy metric the paper reports everywhere, checkpoint I/O, a tiny
CSV logger) that every other script needs. Kept in one file so train.py
and evaluate.py don't duplicate boilerplate.

HOW
---
    from utils import set_seed, balanced_accuracy, CSVLogger, get_device
"""

import os
import csv
import random
import numpy as np
import torch


def set_seed(seed: int):
    """WHY: the paper's few-shot experiments report confidence intervals
    across runs, which only means something if each run is otherwise
    reproducible. Call this once at the top of every script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[utils] Seed set to {seed}")


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[utils] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        if preferred == "cuda":
            print("[utils] CUDA requested but not available -- falling back to CPU. "
                  "Training will be slow; see README for GPU setup notes.")
        else:
            print("[utils] Using CPU.")
    return dev


def balanced_accuracy(y_true, y_pred) -> float:
    """
    WHY: every table in the paper (Table 3, 4, 5) reports "balanced
    accuracy" specifically because TUAB / TUAB-derived tasks are not
    perfectly class-balanced (e.g. gender, medication). Balanced accuracy
    = average of per-class recall, so it isn't dominated by the majority
    class the way plain accuracy would be.

    y_true, y_pred: 1D array-like of 0/1 ints. Samples with y_true == -1
    (our "label unknown / could not be parsed from report" sentinel) are
    dropped before computing the metric, and how many were dropped is
    printed so silent label-extraction failures are always visible.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = y_true != -1
    n_dropped = int((~valid).sum())
    if n_dropped:
        print(f"[utils] balanced_accuracy: dropped {n_dropped} samples "
              f"with unknown label (-1)")
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        print("[utils] balanced_accuracy: no valid labels, returning nan")
        return float("nan")

    accs = []
    for c in np.unique(y_true):
        mask = y_true == c
        if mask.sum() == 0:
            continue
        accs.append((y_pred[mask] == y_true[mask]).mean())
    return float(np.mean(accs))


class CSVLogger:
    """Appends one dict per call to a CSV file, printing every row as it
    writes so training/eval progress is always visible in stdout too.
    If append=False and path exists, saves an automatic timestamped backup
    so each fresh run starts with a clean file while preserving history."""

    def __init__(self, path: str, fieldnames, append: bool = True):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not append and os.path.exists(path) and os.path.getsize(path) > 0:
            import shutil, datetime
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(path)
            backup_path = f"{base}_{mtime}{ext}"
            try:
                shutil.copy2(path, backup_path)
            except Exception:
                pass
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        elif not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        print(f"[utils] Logging to {path}")

    def log(self, row: dict):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)
        print("[log] " + "  ".join(f"{k}={v}" for k, v in row.items()))


def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, path)
    print(f"[utils] Saved checkpoint to {path} (epoch {epoch})")


def load_checkpoint(model, path, optimizer=None, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[utils] Loaded checkpoint from {path} (epoch {ckpt.get('epoch')})")
    return ckpt.get("epoch", 0)
