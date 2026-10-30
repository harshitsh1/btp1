"""
train.py
========
WHY
---
This script runs the contrastive PRE-TRAINING stage of EEG-CLIP / EEG-CLIP++
(paper Sec 3.3): for every (EEG window, report text) pair in the training
split, push the model to make matching pairs' embeddings similar and
mismatched pairs' embeddings dissimilar in the shared 64-dim space.

EEG-CLIP++ extensions:
  - Selects loss function based on cfg.loss_type: "infonce" (original),
    "sigmoid" (SigLIP), or "soft_target".
  - Trains auxiliary structured-label heads jointly when cfg.use_auxiliary_heads
    is True, giving the EEG encoder explicit clinical supervision.
  - Uses the channel-aware encoder when cfg.use_channel_aware_encoder is True.
  - Logs both contrastive and auxiliary loss components separately.

No task labels (pathological/age/gender/medication) are used here at all --
that's the whole point of the approach (task-agnostic representation learning).
Task labels are only used downstream, in evaluate.py.

Hyperparameters (batch size 64, Adam lr=5e-3, weight_decay=5e-4, 20 epochs,
text-encoder LR = 1e-3 x EEG-encoder LR) are exactly the ones reported in
paper Sec 3.3 / Sec 4.3.1, and all live in config.py so you can tweak them
without touching this file.

HOW
---
1. Edit config.py: set `data_root` to your local TUAB copy (and
   `reports_dir` if your report .txt files aren't sitting next to the
   .edf files -- see data/dataset.py docstring for the exact lookup rule).
2. From the project root, on a machine with a CUDA GPU:
       python train.py
   Useful flags:
       python train.py --epochs 5 --batch_size 32          # quick run
       python train.py --text_encoder medicalai/ClinicalBERT
       python train.py --max_recordings 50                 # smoke test
       python train.py --loss_type sigmoid                 # SigLIP loss
       python train.py --no_auxiliary_heads                 # disable aux heads
       python train.py --no_channel_aware                   # use original Deep4
3. Watch stdout: every epoch prints running loss, and a checkpoint +
   CSV log row are written to outputs/checkpoints and outputs/logs after
   every epoch, so you can stop/resume or track progress with e.g.
   `tail -f outputs/logs/train.csv`.
4. The final checkpoint (outputs/checkpoints/eeg_clip_final.pt) is what
   evaluate.py loads for all downstream evaluation.
"""

from config import cfg  # Must be first to redirect all temp and cache dirs to E: drive
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader

from data.dataset import TUABDataset, collate_fn
from models.eeg_clip import EEGCLIP, EEGCLIPPlusPlus, EEGCLIPMultimodal, clip_contrastive_loss
from utils import set_seed, get_device, CSVLogger, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train EEG-CLIP(++) (contrastive pretraining).")
    p.add_argument("--epochs", type=int, default=cfg.n_epochs)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--eeg_lr", type=float, default=cfg.eeg_lr)
    p.add_argument("--text_lr_ratio", type=float, default=cfg.text_lr_ratio)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--text_encoder", type=str, default=cfg.text_encoder_name)
    p.add_argument("--projection_dim", type=int, default=cfg.projection_dim)
    p.add_argument("--text_section_mode", type=str, default=cfg.text_section_mode,
                    help="'all', 'none', or a single section name e.g. IMPRESSION "
                         "(reproduces the ablation in paper Fig. 5)")
    p.add_argument("--max_recordings", type=int, default=None,
                    help="Debug: only load the first N recordings per split.")
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume training from.")
    # EEG-CLIP++ flags
    p.add_argument("--loss_type", type=str, default=cfg.loss_type,
                    choices=["infonce", "sigmoid", "soft_target"],
                    help="Contrastive loss: infonce (original), sigmoid (SigLIP), "
                         "soft_target (text-similarity soft labels)")
    p.add_argument("--no_auxiliary_heads", action="store_true",
                    help="Disable structured-label auxiliary heads")
    p.add_argument("--channel_aware", action="store_true",
                    help="Use channel-aware Transformer encoder instead of Deep4")
    p.add_argument("--no_channel_aware", action="store_true",
                    help="Use original Deep4 encoder instead of channel-aware")
    p.add_argument("--aux_loss_weight", type=float, default=cfg.aux_loss_weight,
                    help="Weight of auxiliary loss relative to contrastive loss")
    # Multimodal (Vision & Signal-to-Text) flags
    p.add_argument("--use_vision", action="store_true",
                    help="Enable Vision branch (3-way EEG + Text + Vision contrastive learning)")
    p.add_argument("--vision_encoder", type=str, default=cfg.vision_encoder_name,
                    help="Vision model backbone: 'resnet18' (default), 'vit_b_16', 'resnet50'")
    p.add_argument("--vision_image_mode", type=str, default=cfg.vision_image_mode,
                    choices=["topomap", "spectrogram", "hybrid"],
                    help="EEG-to-image conversion mode: 'topomap', 'spectrogram', 'hybrid'")
    p.add_argument("--freeze_text_layers", type=int, default=cfg.freeze_text_encoder_layers,
                    help="Number of lower BERT layers to freeze (default 10 to fit in GPU memory)")
    p.add_argument("--no_signal_text", action="store_true",
                    help="Disable dynamic signal-to-text synthesis (use static fallback text)")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch, logger,
                     use_plusplus=True, scaler=None):
    """
    One training epoch. When use_plusplus is True and model is EEGCLIPPlusPlus,
    uses the combined loss (contrastive + auxiliary). Otherwise falls back to
    the original contrastive-only training loop.
    Uses AMP mixed precision for memory efficiency and RTX Tensor Core speedup.
    """
    model.train()
    running_loss = 0.0
    running_contrastive = 0.0
    running_auxiliary = 0.0
    n_batches = 0
    t0 = time.time()
    use_amp = (device.type == "cuda")

    for step, batch in enumerate(loader):
        # Unpack batch (supports 5 items or 6 items if vision is enabled)
        if len(batch) == 6:
            eeg, texts, labels, subject_ids, structured_labels, images = batch
            images = images.to(device, non_blocking=True)
        else:
            eeg, texts, labels, subject_ids, structured_labels = batch
            images = None

        eeg = eeg.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_plusplus and isinstance(model, EEGCLIPPlusPlus):
                # Move structured labels to device
                sl_device = {k: v.to(device) for k, v in structured_labels.items()}
                loss, loss_dict = model.forward_with_loss(eeg, texts, sl_device, images=images)
            else:
                # Original EEG-CLIP training loop
                eeg_z, text_z, logit_scale = model(eeg, texts)
                loss = clip_contrastive_loss(eeg_z, text_z, logit_scale)
                loss_dict = {
                    "total": loss.item(),
                    "contrastive": loss.item(),
                    "auxiliary": 0.0,
                    "logit_scale": logit_scale.item(),
                }

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss_dict["total"]
        running_contrastive += loss_dict["contrastive"]
        running_auxiliary += loss_dict["auxiliary"]
        n_batches += 1

        if step % 20 == 0:
            msg = (f"[train] epoch {epoch} step {step}/{len(loader)} "
                   f"loss={loss_dict['total']:.4f} "
                   f"(contrastive={loss_dict['contrastive']:.4f}, "
                   f"aux={loss_dict['auxiliary']:.4f}) ")
            if "eeg_vision" in loss_dict:
                msg += f"[e<->v={loss_dict['eeg_vision']:.3f}, t<->v={loss_dict['text_vision']:.3f}] "
            msg += (f"logit_scale={loss_dict['logit_scale']:.2f} "
                    f"({time.time() - t0:.1f}s elapsed)")
            print(msg)

    avg_loss = running_loss / max(n_batches, 1)
    avg_contrastive = running_contrastive / max(n_batches, 1)
    avg_auxiliary = running_auxiliary / max(n_batches, 1)

    logger.log({
        "epoch": epoch,
        "avg_total_loss": f"{avg_loss:.5f}",
        "avg_contrastive_loss": f"{avg_contrastive:.5f}",
        "avg_auxiliary_loss": f"{avg_auxiliary:.5f}",
        "seconds": f"{time.time() - t0:.1f}",
    })
    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Sync CLI overrides into cfg so every downstream module sees them.
    cfg.batch_size = args.batch_size
    cfg.eeg_lr = args.eeg_lr
    cfg.text_lr_ratio = args.text_lr_ratio
    cfg.weight_decay = args.weight_decay
    cfg.text_encoder_name = args.text_encoder
    cfg.projection_dim = args.projection_dim
    cfg.text_section_mode = args.text_section_mode
    cfg.device = str(device)

    # EEG-CLIP++ overrides
    cfg.loss_type = args.loss_type
    if args.no_auxiliary_heads:
        cfg.use_auxiliary_heads = False
    if args.channel_aware:
        cfg.use_channel_aware_encoder = True
    elif args.no_channel_aware:
        cfg.use_channel_aware_encoder = False
    cfg.aux_loss_weight = args.aux_loss_weight

    # Multimodal overrides
    if args.use_vision:
        cfg.use_vision_encoder = True
    cfg.vision_encoder_name = args.vision_encoder
    cfg.vision_image_mode = args.vision_image_mode
    cfg.freeze_text_encoder_layers = args.freeze_text_layers
    if args.no_signal_text:
        cfg.use_signal_text = False

    # Determine if we're using EEG-CLIP++ or Multimodal
    use_multimodal = cfg.use_vision_encoder
    use_plusplus = (use_multimodal or
                    cfg.loss_type != "infonce" or
                    cfg.use_auxiliary_heads or
                    cfg.use_channel_aware_encoder)

    print("=" * 70)
    if use_multimodal:
        print("MULTIMODAL EEG-CLIP TRAINING (EEG + Clinical-Text + Vision)")
        print(f"WHY: 3-way contrastive alignment connecting EEG time series,")
        print(f"     signal-derived clinical text, and 2D {cfg.vision_image_mode} maps.")
        print(f"     Vision backbone: {cfg.vision_encoder_name}")
    elif use_plusplus:
        print("EEG-CLIP++ TRAINING")
        print("WHY: learning a shared EEG<->clinical-text embedding space via")
        print("     contrastive learning with SigLIP loss, auxiliary structured")
        print("     heads, and channel-aware EEG encoding.")
    else:
        print("EEG-CLIP TRAINING (original)")
        print("WHY: learning a shared EEG<->clinical-text embedding space via")
        print("     contrastive learning (InfoNCE).")
    print(f"     Loss: {cfg.loss_type}")
    print(f"     Signal-to-text synthesis: {cfg.use_signal_text}")
    print(f"     Auxiliary heads: {cfg.use_auxiliary_heads}")
    print(f"     Channel-aware encoder: {cfg.use_channel_aware_encoder}")
    print("=" * 70)

    # ---------------------------------------------------------------- data
    print("\n[train] Building training dataset...")
    train_ds = TUABDataset(cfg, split=cfg.train_split_name,
                            max_recordings=args.max_recordings)
    if len(train_ds) == 0:
        raise RuntimeError(
            "Training dataset is empty. Check cfg.data_root in config.py "
            "-- it must point at a directory containing "
            "edf/train/{normal,abnormal}/**/*.edf as described in "
            "AAREADME.txt."
        )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        drop_last=True,  # keeps every contrastive batch full-size (paper's
                          # in-batch negatives assumption)
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------------- model
    if use_multimodal:
        print("\n[train] Building Multimodal EEG-CLIP model...")
        model = EEGCLIPMultimodal(cfg).to(device)
    elif use_plusplus:
        print("\n[train] Building EEG-CLIP++ model...")
        model = EEGCLIPPlusPlus(cfg).to(device)
    else:
        print("\n[train] Building EEG-CLIP model...")
        model = EEGCLIP(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model built. Total params: {n_params:,} "
          f"(trainable: {n_trainable:,})")

    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.eeg_lr, cfg.text_lr_ratio, cfg.weight_decay)
    )

    start_epoch = 0
    if args.resume:
        from utils import load_checkpoint
        start_epoch = load_checkpoint(model, args.resume, optimizer,
                                       map_location=device) + 1

    logger = CSVLogger(os.path.join(cfg.log_dir, "train.csv"),
                        fieldnames=["epoch", "avg_total_loss",
                                    "avg_contrastive_loss",
                                    "avg_auxiliary_loss", "seconds"])

    # ------------------------------------------------------------- training
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    print(f"\n[train] Starting training: {args.epochs} epochs, "
          f"batch_size={cfg.batch_size}, {len(train_loader)} steps/epoch\n")

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device,
                                    epoch, logger, use_plusplus=use_plusplus,
                                    scaler=scaler)
        print(f"[train] === Epoch {epoch} done. Avg loss: "
              f"{avg_loss:.4f} ===\n")
        save_checkpoint(model, optimizer, epoch,
                         os.path.join(cfg.checkpoint_dir,
                                      f"eeg_clip_epoch{epoch}.pt"))

    final_path = os.path.join(cfg.checkpoint_dir, "eeg_clip_final.pt")
    save_checkpoint(model, optimizer, args.epochs - 1, final_path)
    print(f"\n[train] Training complete. Final checkpoint: {final_path}")
    print("[train] Next step: run `python evaluate.py` to measure "
          "classification, zero-shot and few-shot performance "
          "(paper Sec 3.4 / Tables 3-5).")


if __name__ == "__main__":
    main()
