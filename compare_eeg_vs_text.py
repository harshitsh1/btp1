"""
compare_eeg_vs_text.py
=======================
WHY
---
The single most important question about EEG-CLIP is: "does adding text
supervision during pretraining actually help, compared to just using the
EEG signal alone?" evaluate.py answers this, but the result is spread
across a CSV with several other baselines (alternative-task-transfer,
zero-shot, few-shot). This script pulls out exactly that one comparison
and prints/saves it as a clean side-by-side table + bar chart, for every
task:

    EEG-only  (Deep4 CNN trained end-to-end DIRECTLY on the task's labels,
               no text ever involved -- this is the paper's
               "Task-specific" baseline, Sec 3.4.1)
        vs
    EEG+Text  (EEG-CLIP: Deep4 CNN pretrained with the contrastive
               EEG<->report loss, then FROZEN, with only a cheap
               classifier probe trained on top -- Sec 3.4.1's main result)

Both use the exact same EEG encoder architecture (models/eeg_encoder.py),
so any accuracy gap is attributable to the *pretraining signal*
(labels-only vs. text-contrastive), not to model capacity.

HOW
---
    python compare_eeg_vs_text.py --checkpoint outputs/checkpoints/eeg_clip_final.pt

Useful flags:
    python compare_eeg_vs_text.py --tasks pathological age
    python compare_eeg_vs_text.py --probe mlp              # or 'logreg'
    python compare_eeg_vs_text.py --epochs 15              # EEG-only training length
    python compare_eeg_vs_text.py --max_recordings 50      # smoke test
    python compare_eeg_vs_text.py --no_plot                # skip matplotlib PNG

Outputs:
    outputs/logs/compare_eeg_vs_text.csv
    outputs/eeg_vs_text_comparison.png   (bar chart, unless --no_plot)
and everything printed to stdout as a formatted table.
"""

from config import cfg  # Must be first to redirect all temp and cache dirs to E: drive
import argparse
import os
import torch
from data.dataset import TUABDataset
from models.eeg_clip import EEGCLIP, EEGCLIPPlusPlus, EEGCLIPMultimodal
from utils import set_seed, get_device, CSVLogger, load_checkpoint

# Reuse the exact same functions evaluate.py uses, so results are
# guaranteed consistent between the two scripts.
from evaluate import (
    TASK_LABEL_KEY,
    extract_embeddings,
    run_classification_probe,
    train_task_specific_deep4,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare EEG-only vs EEG+Text (EEG-CLIP) accuracy, per task.")
    p.add_argument("--checkpoint", type=str,
                    default=os.path.join(cfg.checkpoint_dir, "eeg_clip_final.pt"))
    p.add_argument("--tasks", type=str, nargs="+", default=cfg.eval_tasks)
    p.add_argument("--probe", type=str, default="mlp", choices=["logreg", "mlp"],
                    help="Classifier probe trained on frozen EEG-CLIP embeddings.")
    p.add_argument("--epochs", type=int, default=10,
                    help="Epochs for the EEG-only task-specific baseline.")
    p.add_argument("--max_recordings", type=int, default=None)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--no_channel_aware", action="store_true",
                    help="Use original Deep4 encoder (override config)")
    p.add_argument("--no_plot", action="store_true")
    return p.parse_args()


def print_table(rows):
    print("\n" + "=" * 66)
    print(f"{'Task':<14}{'EEG-only':>14}{'EEG+Text':>14}{'Gap (Text-only)':>20}")
    print("-" * 66)
    for r in rows:
        gap = r["eeg_text"] - r["eeg_only"]
        sign = "+" if gap >= 0 else ""
        print(f"{r['task']:<14}{r['eeg_only']:>13.3f} {r['eeg_text']:>13.3f} "
              f"{sign}{gap:>18.3f}")
    print("=" * 66)
    print("Positive gap = text supervision helped. Metric: balanced accuracy.\n")


def make_plot(rows, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[compare] matplotlib not installed, skipping plot "
              "(pip install matplotlib to enable).")
        return

    tasks = [r["task"] for r in rows]
    eeg_only = [r["eeg_only"] for r in rows]
    eeg_text = [r["eeg_text"] for r in rows]

    x = range(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], eeg_only, width, label="EEG-only")
    ax.bar([i + width / 2 for i in x], eeg_text, width, label="EEG+Text (EEG-CLIP)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance")
    ax.set_title("EEG-only vs EEG+Text pretraining")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[compare] Saved bar chart to {out_path}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if args.no_channel_aware:
        cfg.use_channel_aware_encoder = False

    # Check whether checkpoint is EEGCLIPMultimodal, EEGCLIPPlusPlus or baseline EEGCLIP
    use_multimodal = cfg.use_vision_encoder
    use_plusplus = (use_multimodal or
                    cfg.loss_type != "infonce" or
                    cfg.use_auxiliary_heads or
                    cfg.use_channel_aware_encoder)
    if os.path.exists(args.checkpoint):
        try:
            ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            keys = ckpt_peek.get("model_state", {}).keys()
            if any("temporal_backbone" in k for k in keys):
                cfg.use_channel_aware_encoder = True
                use_plusplus = True
            elif any("temporal_conv" in k for k in keys):
                cfg.use_channel_aware_encoder = False
            if any("auxiliary_heads" in k for k in keys):
                use_plusplus = True
            if any("sigmoid_bias" in k for k in keys):
                cfg.loss_type = "sigmoid"
            if any("vision_encoder" in k for k in keys):
                cfg.use_vision_encoder = True
                use_multimodal = True
                use_plusplus = True
        except Exception:
            pass

    label_tag = "Multimodal EEG-CLIP" if use_multimodal else ("EEG-CLIP++" if use_plusplus else "EEG-CLIP")
    print("=" * 66)
    print(f"EEG-ONLY  vs  EEG+TEXT ({label_tag})  --  accuracy comparison")
    print("Architecture comparison; evaluating pretraining signal transfer.")
    print("=" * 66)

    print("\n[compare] Loading datasets...")
    train_ds = TUABDataset(cfg, split=cfg.train_split_name,
                            max_recordings=args.max_recordings)
    eval_ds = TUABDataset(cfg, split=cfg.eval_split_name,
                           max_recordings=args.max_recordings)

    print(f"\n[compare] Loading {label_tag} checkpoint (for the EEG+Text side)...")
    if use_multimodal:
        model = EEGCLIPMultimodal(cfg).to(device)
    elif use_plusplus:
        model = EEGCLIPPlusPlus(cfg).to(device)
    else:
        model = EEGCLIP(cfg).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location=device)
    else:
        print(f"[compare] WARNING: checkpoint {args.checkpoint} not found -- "
              f"EEG+Text numbers below will reflect a RANDOM, untrained "
              f"encoder. Run train.py first for meaningful results.")
    model.eval()

    print("\n[compare] Extracting EEG-CLIP embeddings (frozen encoder)...")
    train_z, train_labels, _ = extract_embeddings(model, train_ds, device)
    eval_z, eval_labels, _ = extract_embeddings(model, eval_ds, device)

    logger = CSVLogger(
        os.path.join(cfg.log_dir, "compare_eeg_vs_text.csv"),
        fieldnames=["task", "eeg_only", "eeg_text", "gap"],
        append=False)

    rows = []
    for task in args.tasks:
        key = TASK_LABEL_KEY[task]
        print(f"\n[compare] --- Task: {task} ---")

        print(f"[compare] Training EEG-only baseline (Deep4, {args.epochs} "
              f"epochs, no text supervision)...")
        acc_eeg_only = train_task_specific_deep4(
            train_ds, eval_ds, task, device, epochs=args.epochs)

        print(f"[compare] Training {args.probe} probe on frozen EEG+Text "
              f"(EEG-CLIP) embeddings...")
        acc_eeg_text = run_classification_probe(
            train_z, train_labels[key], eval_z, eval_labels[key],
            mode=args.probe)

        gap = acc_eeg_text - acc_eeg_only
        rows.append({"task": task, "eeg_only": acc_eeg_only, "eeg_text": acc_eeg_text})
        logger.log({"task": task, "eeg_only": f"{acc_eeg_only:.3f}",
                    "eeg_text": f"{acc_eeg_text:.3f}", "gap": f"{gap:+.3f}"})

    print_table(rows)

    if not args.no_plot:
        make_plot(rows, os.path.join(cfg.output_dir, "eeg_vs_text_comparison.png"))

    print("[compare] Done. See outputs/logs/compare_eeg_vs_text.csv "
          "and outputs/eeg_vs_text_comparison.png")


if __name__ == "__main__":
    main()
