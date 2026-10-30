"""
evaluate.py
===========
WHY
---
A contrastively pretrained EEG-CLIP(++) checkpoint is only useful if it
produces embeddings that are actually informative. The paper evaluates
this in three complementary ways (Sec 3.4), all reproduced here:

  1. CLASSIFICATION (Sec 3.4.1, Table 3): freeze the pretrained EEG
     encoder, extract embeddings for every window, train a cheap
     downstream classifier (Logistic Regression AND a small MLP) on top,
     using the FULL label set. Compared against:
       - "Task-specific": a Deep4 CNN trained end-to-end from scratch
         directly on the target task (upper bound -- it can use all its
         capacity for exactly one task).
       - "Alternative task transfer": a Deep4 CNN trained end-to-end on
         a DIFFERENT task, then frozen and given a new head for the
         target task (lower bound -- tests whether *any* EEG pretraining
         transfers, without text supervision).

  2. ZERO-SHOT (Sec 3.4.2, Table 4): no classifier training at all --
     classify purely by which of two class-prompt text embeddings (Table 2
     in the paper, cfg.zero_shot_prompts here) each EEG embedding is
     closer to in the shared space.

     **EEG-CLIP++ extension**: optionally train CoOp-style learned prompt
     embeddings on a small validation set, then use those for zero-shot
     instead of hand-written prompts (cfg.use_learned_prompts).

  3. FEW-SHOT (Sec 3.4.3, Table 5 / Fig 4): hold out 20% of the TRAIN
     split, train tiny classifiers on shrinking fractions of it
     (1/2 ... 1/50), and see how gracefully each method degrades.

Every task's metric is BALANCED ACCURACY (see utils.balanced_accuracy),
matching every table in the paper.

HOW
---
    python evaluate.py --checkpoint outputs/checkpoints/eeg_clip_final.pt

Useful flags:
    python evaluate.py --skip_baselines      # EEG-CLIP results only, fast
    python evaluate.py --tasks pathological age
    python evaluate.py --max_recordings 50   # smoke test
    python evaluate.py --use_learned_prompts # CoOp prompts instead of fixed

Outputs:
    outputs/logs/eval_classification.csv
    outputs/logs/eval_zeroshot.csv
    outputs/logs/eval_fewshot.csv
  plus everything printed to stdout as it's computed.
"""

from config import cfg  # Must be first to redirect all temp and cache dirs to E: drive
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from data.dataset import TUABDataset, collate_fn
from data.structured_labels import STRUCTURED_LABEL_NAMES
from models.eeg_clip import EEGCLIP, EEGCLIPPlusPlus, EEGCLIPMultimodal
from models.eeg_encoder import Deep4EEGEncoder
from utils import set_seed, get_device, balanced_accuracy, CSVLogger, load_checkpoint

TASK_LABEL_KEY = {
    "pathological": "pathological",
    "age": "age_binary",
    "gender": "gender_binary",
    "medication": "medication",
}


# =============================================================================
# Embedding extraction
# =============================================================================

@torch.no_grad()
def extract_embeddings(model, dataset: TUABDataset, device,
                        batch_size=128):
    """
    Runs the frozen EEG-CLIP EEG encoder over every window in `dataset`
    and collects (embedding, all 4 labels, subject_id) for downstream
    sklearn classifiers / zero-shot cosine similarity.

    WHY a separate pass instead of reusing train.py's loader: downstream
    evaluation needs embeddings paired with LABELS (which training deliberately
    never looks at), and typically wants them as one big in-memory numpy
    array for sklearn, not streamed minibatches.
    """
    model.eval()

    # Temporarily disable CPU-heavy topomap and signal-text generation during embedding extraction.
    # extract_embeddings only passes raw EEG to model.encode_eeg(eeg), so generating 224x224
    # topomaps or text on CPU for 57k+ windows wastes 10-15 minutes of compute time.
    old_vision = getattr(cfg, "use_vision_encoder", False)
    old_signal_text = getattr(cfg, "use_signal_text", False)
    cfg.use_vision_encoder = False
    cfg.use_signal_text = False

    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=cfg.num_workers)

        all_z, all_labels, all_subjects = [], {k: [] for k in TASK_LABEL_KEY.values()}, []
        n_batches = len(loader)

        for step, batch in enumerate(loader):
            eeg, texts, labels, subject_ids, structured_labels = batch[:5]
            eeg = eeg.to(device)
            z = model.encode_eeg(eeg).cpu().numpy()
            all_z.append(z)
            for k in TASK_LABEL_KEY.values():
                all_labels[k].append(labels[k].numpy())
            all_subjects.extend(subject_ids)

            if (step + 1) % 50 == 0 or (step + 1) == n_batches:
                processed_windows = min((step + 1) * batch_size, len(dataset))
                print(f"  [extract:{dataset.split}] Batch {step + 1}/{n_batches} "
                      f"({processed_windows}/{len(dataset)} windows)...", flush=True)

        all_z = np.concatenate(all_z, axis=0)
        all_labels = {k: np.concatenate(v, axis=0) for k, v in all_labels.items()}
        print(f"[evaluate] Extracted {all_z.shape[0]} embeddings of dim "
              f"{all_z.shape[1]} from '{dataset.split}' split.", flush=True)
        return all_z, all_labels, np.array(all_subjects)
    finally:
        cfg.use_vision_encoder = old_vision
        cfg.use_signal_text = old_signal_text


# =============================================================================
# 1. Classification (paper Sec 3.4.1, Table 3)
# =============================================================================

def run_classification_probe(train_z, train_y, eval_z, eval_y, mode="logreg"):
    """
    WHY: cheap linear (LogReg) and mildly non-linear (3-layer MLP) probes
    on FROZEN embeddings are the standard way to measure representation
    quality without letting the classifier do all the work (paper Sec
    3.4.1 compares both, Table 3).
    """
    valid_tr = train_y != -1
    valid_ev = eval_y != -1
    if valid_tr.sum() == 0 or valid_ev.sum() == 0 or len(np.unique(train_y[valid_tr])) < 2:
        print(f"[evaluate] Skipping {mode} probe: not enough labeled data.")
        return float("nan")

    if mode == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif mode == "mlp":
        use_early_stop = (valid_tr.sum() >= 25)
        clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                             early_stopping=use_early_stop, random_state=cfg.seed)
    else:
        raise ValueError(mode)

    try:
        clf.fit(train_z[valid_tr], train_y[valid_tr])
        pred = clf.predict(eval_z[valid_ev])
        return balanced_accuracy(eval_y[valid_ev], pred)
    except Exception as e:
        print(f"[evaluate] {mode} probe fit failed ({e}), returning nan.")
        return float("nan")


def train_task_specific_deep4(train_ds, eval_ds, task, device, epochs=10):
    """
    Baseline "Task-specific" (Table 3 upper bound): a Deep4 CNN trained
    end-to-end directly on `task`'s labels, from random initialization.
    Uses the same architecture as EEG-CLIP's EEG encoder so the comparison
    isolates the value of contrastive pretraining, not architecture size.
    """
    label_key = TASK_LABEL_KEY[task]
    print(f"[evaluate] Training task-specific Deep4 baseline for '{task}' "
          f"({epochs} epochs)...")

    # Guard: if the dataset has no items, we can't train.
    if len(train_ds) == 0:
        print(f"[evaluate] Skipping task-specific baseline for '{task}': "
              f"empty training dataset.")
        return float("nan")

    encoder = Deep4EEGEncoder(n_channels=cfg.n_channels,
                               input_window_samples=cfg.window_len_samples,
                               embed_dim=cfg.eeg_embed_dim).to(device)
    head = torch.nn.Linear(cfg.eeg_embed_dim, 2).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=cfg.weight_decay)

    # Temporarily disable vision and signal text during EEG-only baseline training
    # to avoid wasting CPU computing 224x224 topomaps that Deep4 discards.
    old_vision = getattr(cfg, "use_vision_encoder", False)
    old_signal_text = getattr(cfg, "use_signal_text", False)
    cfg.use_vision_encoder = False
    cfg.use_signal_text = False

    try:
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=cfg.num_workers,
                             drop_last=True)

        for epoch in range(epochs):
            encoder.train(); head.train()
            total, n = 0.0, 0
            for batch in loader:
                eeg, texts, labels, _, structured_labels = batch[:5]
                y = labels[label_key]
                valid = y != -1
                if valid.sum() == 0:
                    continue
                eeg, y = eeg[valid].to(device), y[valid].to(device)
                z = encoder(eeg)
                logits = head(z)
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
            if n:
                print(f"  [task-specific:{task}] epoch {epoch} loss={total / n:.4f}")

        res = _eval_encoder_head(encoder, head, eval_ds, label_key, device)
    finally:
        cfg.use_vision_encoder = old_vision
        cfg.use_signal_text = old_signal_text

    return res


def train_alternative_task_transfer(train_ds, eval_ds, source_task, target_task,
                                     device, epochs=10):
    """
    Baseline "Alternative task transfer" (Table 3 lower bound): pretrain a
    Deep4 encoder end-to-end on `source_task`, freeze it, and train only a
    new linear head on `target_task`. Tests whether ANY supervised EEG
    pretraining transfers, without text supervision -- the comparison
    point that makes EEG-CLIP's use of text supervision meaningful.
    """
    print(f"[evaluate] Alternative-task baseline: pretrain on '{source_task}', "
          f"transfer to '{target_task}'")
    source_key = TASK_LABEL_KEY[source_task]

    encoder = Deep4EEGEncoder(n_channels=cfg.n_channels,
                               input_window_samples=cfg.window_len_samples,
                               embed_dim=cfg.eeg_embed_dim).to(device)
    source_head = torch.nn.Linear(cfg.eeg_embed_dim, 2).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(source_head.parameters()),
                            lr=1e-3, weight_decay=cfg.weight_decay)
    old_vision = getattr(cfg, "use_vision_encoder", False)
    old_signal_text = getattr(cfg, "use_signal_text", False)
    cfg.use_vision_encoder = False
    cfg.use_signal_text = False

    try:
        loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=cfg.num_workers,
                             drop_last=True)

        for epoch in range(epochs):
            encoder.train(); source_head.train()
            for batch in loader:
                eeg, texts, labels, _, structured_labels = batch[:5]
                y = labels[source_key]
                valid = y != -1
                if valid.sum() == 0:
                    continue
                eeg, y = eeg[valid].to(device), y[valid].to(device)
                logits = source_head(encoder(eeg))
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()

        # freeze encoder, train new head on target task
        for p in encoder.parameters():
            p.requires_grad = False
        target_head = torch.nn.Linear(cfg.eeg_embed_dim, 2).to(device)
        opt2 = torch.optim.Adam(target_head.parameters(), lr=1e-3)
        target_key = TASK_LABEL_KEY[target_task]
        for epoch in range(epochs):
            encoder.eval(); target_head.train()
            for batch in loader:
                eeg, texts, labels, _, structured_labels = batch[:5]
                y = labels[target_key]
                valid = y != -1
                if valid.sum() == 0:
                    continue
                eeg, y = eeg[valid].to(device), y[valid].to(device)
                with torch.no_grad():
                    z = encoder(eeg)
                logits = target_head(z)
                loss = torch.nn.functional.cross_entropy(logits, y)
                opt2.zero_grad(); loss.backward(); opt2.step()

        res = _eval_encoder_head(encoder, target_head, eval_ds, target_key, device)
    finally:
        cfg.use_vision_encoder = old_vision
        cfg.use_signal_text = old_signal_text

    return res


@torch.no_grad()
def _eval_encoder_head(encoder, head, eval_ds, label_key, device):
    encoder.eval(); head.eval()
    old_vision = getattr(cfg, "use_vision_encoder", False)
    old_signal_text = getattr(cfg, "use_signal_text", False)
    cfg.use_vision_encoder = False
    cfg.use_signal_text = False
    try:
        loader = DataLoader(eval_ds, batch_size=128, shuffle=False,
                             collate_fn=collate_fn, num_workers=cfg.num_workers)
        all_pred, all_true = [], []
        for batch in loader:
            eeg, texts, labels, _, structured_labels = batch[:5]
            y = labels[label_key]
            valid = y != -1
            if valid.sum() == 0:
                continue
            eeg = eeg[valid].to(device)
            pred = head(encoder(eeg)).argmax(dim=1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(y[valid].numpy())
        if not all_pred:
            return float("nan")
        return balanced_accuracy(np.concatenate(all_true), np.concatenate(all_pred))
    finally:
        cfg.use_vision_encoder = old_vision
        cfg.use_signal_text = old_signal_text


# =============================================================================
# 2. Zero-shot classification (paper Sec 3.4.2, Table 4)
# =============================================================================

@torch.no_grad()
def run_zero_shot(model, eval_z: np.ndarray, eval_labels: dict, device):
    """
    For each task, embed its two class prompts (cfg.zero_shot_prompts,
    matching Table 2 of the paper) with the SAME text encoder used during
    contrastive training, then classify every eval-set EEG embedding by
    cosine similarity to the two prompt embeddings -- no label ever used
    for training, only for measuring accuracy afterwards.
    """
    results = {}
    eval_z_t = torch.from_numpy(eval_z).float().to(device)
    eval_z_t = torch.nn.functional.normalize(eval_z_t, dim=-1)

    for task, (prompt_a, prompt_b) in cfg.zero_shot_prompts.items():
        text_z = model.encode_text([prompt_a, prompt_b], device=device)  # (2, D)
        sims = eval_z_t @ text_z.t()          # (N, 2): col0=prompt_a, col1=prompt_b
        pred = sims.argmax(dim=1).cpu().numpy()  # 0 -> class A, 1 -> class B

        y = eval_labels[TASK_LABEL_KEY[task]]
        acc = balanced_accuracy(y, pred)
        results[task] = acc
        print(f"[evaluate:zero-shot] {task:12s} balanced_acc={acc:.3f}  "
              f"prompts=({prompt_a!r}, {prompt_b!r})")
    return results


# =============================================================================
# 2b. Zero-shot with learned prompts (EEG-CLIP++ improvement 4)
# =============================================================================

def train_prompt_learner(model, train_ds, device, task, n_epochs=10,
                          n_val_samples=200):
    """
    Train CoOp-style learned prompt embeddings on a small validation
    subset for a specific task.

    WHY: hand-written prompts like "This is a normal recording" may not
    align with what the model actually learned. Learned prompts adapt to
    the model's embedding space and typically give large zero-shot boosts.

    Uses a small held-out subset (n_val_samples) of the training set for
    prompt optimization. The text encoder stays frozen -- only the M
    learnable prompt tokens are updated.
    """
    from models.prompt_learner import PromptLearner
    import torch.nn.functional as F

    class_names = cfg.prompt_class_names.get(task, ["class_0", "class_1"])
    label_key = TASK_LABEL_KEY[task]

    prompt_learner = PromptLearner(
        text_encoder=model.text_encoder,
        class_names=class_names,
        n_prompt_tokens=cfg.n_prompt_tokens,
        class_specific=cfg.prompt_class_specific,
        prompt_init=cfg.prompt_init_template,
    ).to(device)

    old_vision = getattr(cfg, "use_vision_encoder", False)
    old_signal_text = getattr(cfg, "use_signal_text", False)
    cfg.use_vision_encoder = False
    cfg.use_signal_text = False

    try:
        # Get a small validation subset for prompt training
        loader = DataLoader(train_ds, batch_size=min(64, n_val_samples),
                             shuffle=True, collate_fn=collate_fn,
                             num_workers=cfg.num_workers)

        optimizer = torch.optim.Adam(
            [prompt_learner.prompt_embeddings], lr=cfg.prompt_lr)

        print(f"[evaluate:learned-prompts] Training prompts for '{task}' "
              f"({n_epochs} epochs, {cfg.n_prompt_tokens} tokens, "
              f"class_names={class_names})")

        model.eval()
        prompt_learner.train()

        for epoch in range(n_epochs):
            total_loss = 0.0
            n_batches = 0
            samples_seen = 0

            for batch in loader:
                eeg, texts, labels, _, structured_labels = batch[:5]
                y = labels[label_key]
                valid = y != -1
                if valid.sum() == 0:
                    continue

                eeg = eeg[valid].to(device)
                y = y[valid].to(device)

                with torch.no_grad():
                    eeg_z = model.encode_eeg(eeg)  # (n, D)

                # Get prompted text embeddings
                class_embeds = prompt_learner(device=device)  # (n_classes, H)

                # Project through text projection head
                class_z = model.text_proj(class_embeds)        # (n_classes, proj_dim)
                class_z = F.normalize(class_z, dim=-1)

                # Cosine similarity -> cross-entropy loss
                logits = eeg_z @ class_z.t()  # (n, n_classes)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                samples_seen += valid.sum().item()

                if samples_seen >= n_val_samples:
                    break

            if n_batches > 0 and epoch % 3 == 0:
                print(f"  [prompt-train:{task}] epoch {epoch} "
                      f"loss={total_loss / n_batches:.4f}")

        return prompt_learner
    finally:
        cfg.use_vision_encoder = old_vision
        cfg.use_signal_text = old_signal_text


@torch.no_grad()
def run_zero_shot_learned(model, eval_z: np.ndarray, eval_labels: dict,
                           device, prompt_learners: dict):
    """
    Zero-shot classification using learned prompt embeddings.
    Same as run_zero_shot but uses PromptLearner outputs instead of
    fixed text prompts.
    """
    import torch.nn.functional as F

    results = {}
    eval_z_t = torch.from_numpy(eval_z).float().to(device)
    eval_z_t = F.normalize(eval_z_t, dim=-1)

    for task, learner in prompt_learners.items():
        learner.eval()

        # Get learned class embeddings
        class_embeds = learner(device=device)  # (n_classes, H)
        class_z = model.text_proj(class_embeds)  # (n_classes, proj_dim)
        class_z = F.normalize(class_z, dim=-1)

        sims = eval_z_t @ class_z.t()  # (N, n_classes)
        pred = sims.argmax(dim=1).cpu().numpy()

        y = eval_labels[TASK_LABEL_KEY[task]]
        acc = balanced_accuracy(y, pred)
        results[task] = acc
        class_names = cfg.prompt_class_names.get(task, ["?", "?"])
        print(f"[evaluate:learned-zero-shot] {task:12s} balanced_acc={acc:.3f}  "
              f"classes={class_names}")

    return results


# =============================================================================
# 3. Few-shot classification (paper Sec 3.4.3, Table 5 / Fig 4)
# =============================================================================

def run_few_shot(model, train_ds, eval_ds, device, tasks, fractions,
                  run_baselines=True):
    """
    WHY: splits the TRAIN recordings by SUBJECT (never by window) into a
    60% "already used for contrastive pretraining" pool and a 20%
    few-shot pool, matching Fig. 3's partitioning -- subject-level
    splitting avoids leaking a subject's other windows across the split.
    We then train classifiers on shrinking fractions of the 20% pool and
    evaluate on the official eval split, for EEG-CLIP-frozen-embeddings,
    task-specific-from-scratch, and (optionally) alternative-task-transfer.
    """
    rng = np.random.RandomState(cfg.seed)
    all_subjects = sorted(set(it["subject_id"] for it in train_ds.items))
    rng.shuffle(all_subjects)
    n_fewshot_subj = int(len(all_subjects) * cfg.few_shot_holdout_frac)
    fewshot_subjects = set(all_subjects[:n_fewshot_subj])
    contrastive_subjects = set(all_subjects[n_fewshot_subj:])
    print(f"[evaluate:few-shot] {len(contrastive_subjects)} subjects reserved "
          f"for contrastive pretraining, {len(fewshot_subjects)} subjects "
          f"held out for few-shot experiments.")

    if len(fewshot_subjects) == 0:
        print("[evaluate:few-shot] Not enough subjects to create a few-shot holdout pool. Skipping few-shot.")
        return []

    fewshot_pool = TUABDataset(cfg, split=train_ds.split,
                                subject_filter=fewshot_subjects, verbose=False)
    pool_z, pool_labels, pool_subjects = extract_embeddings(model, fewshot_pool, device)

    eval_z, eval_labels, _ = extract_embeddings(model, eval_ds, device)

    results = []
    for task in tasks:
        label_key = TASK_LABEL_KEY[task]
        y_pool = pool_labels[label_key]
        y_eval = eval_labels[label_key]
        valid_pool_idx = np.where(y_pool != -1)[0]

        if len(valid_pool_idx) == 0:
            print(f"[evaluate:few-shot] Skipping task='{task}': "
                  f"no valid labels in pool.")
            continue

        for frac in fractions:
            n_take = max(2, int(len(valid_pool_idx) * frac))
            idx = rng.choice(valid_pool_idx, size=min(n_take, len(valid_pool_idx)),
                              replace=False)

            acc_clip = run_classification_probe(
                pool_z[idx], y_pool[idx], eval_z, y_eval, mode="mlp")

            row = {"task": task, "fraction": frac, "n_samples": len(idx),
                   "eeg_clip_mlp": acc_clip}

            if run_baselines:
                subj_subset = set(pool_subjects[idx])
                small_train_ds = TUABDataset(cfg, split=train_ds.split,
                                              subject_filter=subj_subset,
                                              verbose=False)
                row["task_specific"] = train_task_specific_deep4(
                    small_train_ds, eval_ds, task, device, epochs=5)

            print(f"[evaluate:few-shot] task={task} frac={frac:.3f} "
                  f"n={len(idx)} -> {row}")
            results.append(row)
    return results


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained EEG-CLIP(++) checkpoint.")
    p.add_argument("--checkpoint", type=str,
                    default=os.path.join(cfg.checkpoint_dir, "eeg_clip_final.pt"))
    p.add_argument("--tasks", type=str, nargs="+", default=cfg.eval_tasks)
    p.add_argument("--skip_baselines", action="store_true",
                    help="Skip task-specific / alternative-task baselines "
                         "(much faster, EEG-CLIP results only).")
    p.add_argument("--skip_fewshot", action="store_true")
    p.add_argument("--max_recordings", type=int, default=None)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--seed", type=int, default=cfg.seed)
    # EEG-CLIP++ flags
    p.add_argument("--use_learned_prompts", action="store_true",
                    help="Use CoOp-style learned prompts for zero-shot "
                         "instead of hand-written prompts.")
    p.add_argument("--no_learned_prompts", action="store_true",
                    help="Disable learned prompts and use hand-written prompts.")
    p.add_argument("--channel_aware", action="store_true",
                    help="Use channel-aware Transformer encoder (override config)")
    p.add_argument("--no_channel_aware", action="store_true",
                    help="Use original Deep4 encoder (override config)")
    p.add_argument("--use_vision", action="store_true",
                    help="Enable vision branch for multimodal checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Apply overrides
    if args.no_channel_aware:
        cfg.use_channel_aware_encoder = False
    elif args.channel_aware:
        cfg.use_channel_aware_encoder = True

    if args.use_learned_prompts:
        cfg.use_learned_prompts = True
    elif args.no_learned_prompts:
        cfg.use_learned_prompts = False

    if args.use_vision:
        cfg.use_vision_encoder = True

    # Auto-detect encoder architecture and ++ features from checkpoint if available
    if os.path.exists(args.checkpoint):
        try:
            ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            keys = ckpt_peek.get("model_state", {}).keys()
            if not args.channel_aware and not args.no_channel_aware:
                if any("temporal_backbone" in k for k in keys):
                    cfg.use_channel_aware_encoder = True
                elif any("temporal_conv" in k for k in keys):
                    cfg.use_channel_aware_encoder = False
            if any("sigmoid_bias" in k for k in keys):
                cfg.loss_type = "sigmoid"
            if any("auxiliary_heads" in k for k in keys):
                cfg.use_auxiliary_heads = True
            if any("vision_encoder" in k for k in keys):
                cfg.use_vision_encoder = True
        except Exception:
            pass

    # Determine model class
    use_multimodal = cfg.use_vision_encoder
    use_plusplus = (use_multimodal or
                    cfg.loss_type != "infonce" or
                    cfg.use_auxiliary_heads or
                    cfg.use_channel_aware_encoder)

    print("=" * 70)
    if use_multimodal:
        print("MULTIMODAL EEG-CLIP EVALUATION (EEG + Text + Vision)")
    elif use_plusplus:
        print("EEG-CLIP++ EVALUATION")
    else:
        print("EEG-CLIP EVALUATION")
    print("Reproducing paper Sec 3.4: classification (Table 3), zero-shot")
    print("(Table 4), few-shot (Table 5 / Fig 4). Metric: balanced accuracy.")
    if use_plusplus:
        print(f"  Loss type: {cfg.loss_type}")
        print(f"  Channel-aware encoder: {cfg.use_channel_aware_encoder}")
        print(f"  Auxiliary heads: {cfg.use_auxiliary_heads}")
        print(f"  Learned prompts: {cfg.use_learned_prompts}")
        print(f"  Vision encoder: {cfg.use_vision_encoder}")
    print("=" * 70)

    print("\n[evaluate] Loading datasets...")
    train_ds = TUABDataset(cfg, split=cfg.train_split_name,
                            max_recordings=args.max_recordings)
    eval_ds = TUABDataset(cfg, split=cfg.eval_split_name,
                           max_recordings=args.max_recordings)

    print("\n[evaluate] Loading model + checkpoint...")
    if use_multimodal:
        model = EEGCLIPMultimodal(cfg).to(device)
    elif use_plusplus:
        model = EEGCLIPPlusPlus(cfg).to(device)
    else:
        model = EEGCLIP(cfg).to(device)

    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location=device)
    else:
        print(f"[evaluate] WARNING: checkpoint {args.checkpoint} not found. "
              f"Evaluating a RANDOMLY INITIALIZED model (sanity-check only, "
              f"run train.py first for meaningful numbers).")
    model.eval()

    # ---------------------------------------------------------- embeddings
    print("\n[evaluate] Extracting embeddings...")
    train_z, train_labels, _ = extract_embeddings(model, train_ds, device)
    eval_z, eval_labels, _ = extract_embeddings(model, eval_ds, device)

    # ---------------------------------------------------------- 1. classification
    print("\n" + "-" * 70)
    print("1) CLASSIFICATION (paper Table 3)")
    print("-" * 70)
    clf_logger = CSVLogger(
        os.path.join(cfg.log_dir, "eval_classification.csv"),
        fieldnames=["task", "eeg_clip_logreg", "eeg_clip_mlp",
                    "task_specific", "alternative_task"],
        append=False)
    for task in args.tasks:
        key = TASK_LABEL_KEY[task]
        acc_lr = run_classification_probe(train_z, train_labels[key],
                                           eval_z, eval_labels[key], "logreg")
        acc_mlp = run_classification_probe(train_z, train_labels[key],
                                            eval_z, eval_labels[key], "mlp")
        row = {"task": task, "eeg_clip_logreg": f"{acc_lr:.3f}",
               "eeg_clip_mlp": f"{acc_mlp:.3f}",
               "task_specific": "", "alternative_task": ""}
        if not args.skip_baselines:
            row["task_specific"] = f"{train_task_specific_deep4(train_ds, eval_ds, task, device):.3f}"
            other_task = next(t for t in args.tasks if t != task)
            row["alternative_task"] = f"{train_alternative_task_transfer(train_ds, eval_ds, other_task, task, device):.3f}"
        clf_logger.log(row)

    # ---------------------------------------------------------- 2. zero-shot
    print("\n" + "-" * 70)
    print("2) ZERO-SHOT CLASSIFICATION (paper Table 4)")
    print("-" * 70)

    # 2a. Fixed prompts (original)
    zs_logger = CSVLogger(os.path.join(cfg.log_dir, "eval_zeroshot.csv"),
                           fieldnames=["task", "method", "balanced_accuracy"],
                           append=False)
    zs_results = run_zero_shot(model, eval_z, eval_labels, device)
    for task, acc in zs_results.items():
        zs_logger.log({"task": task, "method": "fixed_prompt",
                        "balanced_accuracy": f"{acc:.3f}"})

    # 2b. Learned prompts (EEG-CLIP++)
    if cfg.use_learned_prompts:
        print("\n" + "-" * 70)
        print("2b) ZERO-SHOT WITH LEARNED PROMPTS (CoOp-style)")
        print("-" * 70)
        prompt_learners = {}
        for task in args.tasks:
            learner = train_prompt_learner(
                model, train_ds, device, task,
                n_epochs=10, n_val_samples=500)
            prompt_learners[task] = learner

        zs_learned_results = run_zero_shot_learned(
            model, eval_z, eval_labels, device, prompt_learners)
        for task, acc in zs_learned_results.items():
            zs_logger.log({"task": task, "method": "learned_prompt",
                            "balanced_accuracy": f"{acc:.3f}"})

    # ---------------------------------------------------------- 3. few-shot
    if not args.skip_fewshot:
        print("\n" + "-" * 70)
        print("3) FEW-SHOT CLASSIFICATION (paper Table 5 / Fig 4)")
        print("-" * 70)
        fs_logger = CSVLogger(
            os.path.join(cfg.log_dir, "eval_fewshot.csv"),
            fieldnames=["task", "fraction", "n_samples", "eeg_clip_mlp",
                        "task_specific"],
            append=False)
        fs_results = run_few_shot(model, train_ds, eval_ds, device,
                                   tasks=args.tasks,
                                   fractions=cfg.few_shot_fractions,
                                   run_baselines=not args.skip_baselines)
        for row in fs_results:
            fs_logger.log({k: row.get(k, "") for k in
                            ["task", "fraction", "n_samples", "eeg_clip_mlp", "task_specific"]})

    print("\n[evaluate] Done. See outputs/logs/eval_*.csv for full results.")


if __name__ == "__main__":
    main()
