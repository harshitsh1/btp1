"""
save_spectrograms.py
====================
WHY
---
Generate and save 2D time-frequency STFT spectrograms (and scalp topomaps)
for each EDF recording in the TUAB dataset, strictly stored on E: drive.

Supports:
  1. Full-recording high-resolution STFT overview (Left, Midline, Right hemisphere
     time-frequency plots across the entire recording with frequency 0-35 Hz and time axes).
  2. Per-window 224x224 RGB image representations (used by Vision Transformer / ResNet-18).
  3. Side-by-side normal vs abnormal spectrogram gallery comparisons.

HOW
---
Run for first 5 recordings (smoke test):
    python save_spectrograms.py --max_recordings 5

Run for full evaluation split:
    python save_spectrograms.py --split eval

Run for training split:
    python save_spectrograms.py --split train --max_recordings 50

Save both spectrograms and scalp topomaps:
    python save_spectrograms.py --mode all
"""

from config import cfg  # Must be first to redirect all temp and cache dirs to E: drive
import argparse
import os
import glob
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data.edf_utils import load_and_preprocess_edf, make_windows
from data.eeg_to_image import (
    eeg_to_image,
    save_image_to_file,
    compute_full_recording_spectrogram,
    denormalize_imagenet,
)


def parse_args():
    p = argparse.ArgumentParser(description="Save spectrograms and topomaps for TUAB EDF files.")
    p.add_argument("--split", type=str, default="eval", choices=["train", "eval", "both"],
                   help="Dataset split to process: 'eval' (default, 276 files), 'train', or 'both'")
    p.add_argument("--mode", type=str, default="spectrogram",
                   choices=["spectrogram", "topomap", "hybrid", "all"],
                   help="Image representation mode: 'spectrogram', 'topomap', 'hybrid', or 'all'")
    p.add_argument("--max_recordings", type=int, default=None,
                   help="Max recordings to process (default: None = all recordings in split)")
    p.add_argument("--output_dir", type=str, default="E:/eeg_clip/outputs/spectrograms",
                   help="Directory to save spectrograms (strictly on E: drive)")
    p.add_argument("--max_windows_per_rec", type=int, default=3,
                   help="Max windows to save per recording (0 = save all windows)")
    p.add_argument("--no_full_recording", action="store_true",
                   help="Skip full recording multi-minute overview spectrograms")
    p.add_argument("--gallery", action="store_true", default=True,
                   help="Create comparison gallery of normal vs abnormal spectrograms")
    return p.parse_args()


def scan_edf_files(data_root, edf_subdir, split, max_recordings=None):
    """Finds all normal and abnormal EDF files for a given split."""
    recordings = []
    needed = max(2, max_recordings // 2 + 1) if max_recordings is not None else None
    for label_name, pathological in [("normal", 0), ("abnormal", 1)]:
        dir_path = os.path.join(data_root, edf_subdir, split, label_name)
        if not os.path.isdir(dir_path):
            continue
        if needed is not None:
            count = 0
            for root, _, files in os.walk(dir_path):
                for fname in sorted(files):
                    if fname.lower().endswith(".edf"):
                        path = os.path.join(root, fname)
                        recordings.append({
                            "edf_path": path,
                            "split": split,
                            "label_name": label_name,
                            "pathological": pathological,
                            "stem": os.path.splitext(fname)[0],
                        })
                        count += 1
                        if count >= needed:
                            break
                if count >= needed:
                    break
        else:
            pattern = os.path.join(dir_path, "**", "*.edf")
            matches = sorted(glob.glob(pattern, recursive=True))
            for path in matches:
                recordings.append({
                    "edf_path": path,
                    "split": split,
                    "label_name": label_name,
                    "pathological": pathological,
                    "stem": os.path.splitext(os.path.basename(path))[0],
                })
    return recordings


def create_gallery(samples, output_dir):
    """Creates a side-by-side visual comparison gallery of Normal vs Abnormal EEG spectrograms."""
    normal_samples = [s for s in samples if s["label_name"] == "normal"]
    abnormal_samples = [s for s in samples if s["label_name"] == "abnormal"]

    if not normal_samples or not abnormal_samples:
        return

    n_show = min(2, len(normal_samples), len(abnormal_samples))
    fig, axes = plt.subplots(n_show * 2, 2, figsize=(10, 3.5 * n_show * 2))

    row = 0
    for i in range(n_show):
        # Normal
        norm_spec = samples[normal_samples[i]["idx"]]["spec_win"]
        norm_topo = samples[normal_samples[i]["idx"]]["topo_win"]
        axes[row, 0].imshow(denormalize_imagenet(norm_spec).transpose(1, 2, 0))
        axes[row, 0].set_title(f"NORMAL EEG: Spectrogram (STFT)\n{normal_samples[i]['stem']}", fontsize=9, fontweight="bold", color="darkgreen")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(denormalize_imagenet(norm_topo).transpose(1, 2, 0))
        axes[row, 1].set_title(f"NORMAL EEG: Scalp Topomap\n{normal_samples[i]['stem']}", fontsize=9, fontweight="bold", color="darkgreen")
        axes[row, 1].axis("off")
        row += 1

        # Abnormal
        abnorm_spec = samples[abnormal_samples[i]["idx"]]["spec_win"]
        abnorm_topo = samples[abnormal_samples[i]["idx"]]["topo_win"]
        axes[row, 0].imshow(denormalize_imagenet(abnorm_spec).transpose(1, 2, 0))
        axes[row, 0].set_title(f"ABNORMAL EEG: Spectrogram (STFT)\n{abnormal_samples[i]['stem']}", fontsize=9, fontweight="bold", color="darkred")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(denormalize_imagenet(abnorm_topo).transpose(1, 2, 0))
        axes[row, 1].set_title(f"ABNORMAL EEG: Scalp Topomap\n{abnormal_samples[i]['stem']}", fontsize=9, fontweight="bold", color="darkred")
        axes[row, 1].axis("off")
        row += 1

    fig.suptitle("TUAB Multimodal Visualizations: Spectrograms & Topomaps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    gallery_path = os.path.join(output_dir, "spectrogram_gallery.png")
    fig.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save_spectrograms] Created visual comparison gallery: {gallery_path}")


def main():
    args = parse_args()
    splits = ["train", "eval"] if args.split == "both" else [args.split]

    print("=" * 70)
    print("TUAB EEG SPECTROGRAM & TOPOMAP GENERATOR")
    print(f"Data root:   {os.path.abspath(cfg.data_root)}")
    print(f"Output dir:  {os.path.abspath(args.output_dir)} (Strictly on E: drive)")
    print(f"Mode:        {args.mode}")
    print(f"Splits:      {splits}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()
    total_saved = 0
    gallery_samples = []

    for split in splits:
        recordings = scan_edf_files(cfg.data_root, cfg.edf_subdir, split, max_recordings=args.max_recordings)
        if args.max_recordings:
            norm = [r for r in recordings if r["label_name"] == "normal"]
            abn = [r for r in recordings if r["label_name"] == "abnormal"]
            half = max(1, args.max_recordings // 2)
            recordings = norm[:half] + abn[:max(1, args.max_recordings - half)]

        print(f"\n[{split}] Processing {len(recordings)} EDF recordings...")

        for i, rec in enumerate(recordings):
            edf_path = rec["edf_path"]
            label_name = rec["label_name"]
            stem = rec["stem"]

            rec_out_dir = os.path.join(args.output_dir, split, label_name, stem)
            os.makedirs(rec_out_dir, exist_ok=True)

            try:
                # Load preprocessed 21-channel EEG recording
                sig = load_and_preprocess_edf(edf_path, cfg)
            except Exception as e:
                print(f"  [skip] Failed to load {stem}: {e}")
                continue

            duration_sec = sig.shape[-1] / cfg.resample_freq

            # 1. Full recording high-resolution overview spectrogram
            if not args.no_full_recording:
                full_spec_path = os.path.join(rec_out_dir, f"{stem}_full_spectrogram.png")
                status_str = "ABNORMAL" if rec["pathological"] else "NORMAL"
                compute_full_recording_spectrogram(
                    sig,
                    fs=cfg.resample_freq,
                    out_path=full_spec_path,
                    title=f"Full Spectrogram: {stem} ({status_str}, {duration_sec:.1f}s)",
                )
                total_saved += 1

            # 2. Slice 12-second windows
            windows = make_windows(sig, cfg.window_len_samples, cfg.window_stride_samples)
            if len(windows) == 0:
                continue

            max_wins = len(windows) if args.max_windows_per_rec == 0 else min(args.max_windows_per_rec, len(windows))

            # Sample for gallery comparison
            win0 = windows[0]
            spec_w0 = eeg_to_image(win0, mode="spectrogram", fs=cfg.resample_freq, img_size=224)
            topo_w0 = eeg_to_image(win0, mode="topomap", fs=cfg.resample_freq, img_size=224)

            if len(gallery_samples) < 10:
                gallery_samples.append({
                    "idx": len(gallery_samples),
                    "label_name": label_name,
                    "stem": stem,
                    "spec_win": spec_w0,
                    "topo_win": topo_w0,
                })

            # Save windows per mode
            modes_to_save = ["spectrogram", "topomap"] if args.mode == "all" else [args.mode]
            for w_idx in range(max_wins):
                w_sig = windows[w_idx]
                for m in modes_to_save:
                    img = eeg_to_image(w_sig, mode=m, fs=cfg.resample_freq, img_size=224, normalize_imagenet=True)
                    win_img_path = os.path.join(rec_out_dir, f"win_{w_idx:02d}_{m}.png")
                    save_image_to_file(img, win_img_path, was_normalized=True)
                    total_saved += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(recordings):
                print(f"  [{split}] ({i + 1}/{len(recordings)}) {stem} ({label_name}) -> saved spectrograms to {rec_out_dir}")

    # Build gallery if requested
    if args.gallery and gallery_samples:
        create_gallery(gallery_samples, args.output_dir)

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"Spectrogram export complete in {elapsed:.1f}s!")
    print(f"Total image files saved: {total_saved}")
    print(f"Saved directory: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
