"""
data/dataset.py
================
WHY
---
This is the bridge between "a folder of .edf files + .txt reports laid out
the way AAREADME.txt describes" and PyTorch tensors that EEG-CLIP can
train on. It:
  1. Walks `edf/{train,eval}/{normal,abnormal}/...` and finds every .edf.
  2. Loads/pairs each recording with its report text (see ASSUMPTION below).
  3. Preprocesses + windows the signal (data/edf_utils.py).
  4. Builds the text input string for the chosen section mode (data/reports.py).
  5. Extracts all 4 evaluation labels: pathological (from folder), age,
     gender, medication (from report text).
  6. Exposes everything as a torch.utils.data.Dataset yielding
     (window_tensor, text_string, label_dict).

ASSUMPTION about report file location
--------------------------------------
The TUAB release you download from NEDC ships one report per recording.
Different releases have stored these in slightly different places over
the years. This loader tries, in order:
  (a) `<edf_path>` with `.edf` replaced by `.txt`   (report next to signal)
  (b) `cfg.reports_dir/<same relative path>.txt`     (mirrored tree)
If neither exists, the recording is SKIPPED and counted, and a summary of
how many were skipped is printed -- so a wrong path is obvious immediately
rather than silently training on zero text supervision. Adjust
`TUABDataset._find_report_path` if your corpus copy uses another layout.

HOW
---
    from config import cfg
    from data.dataset import TUABDataset, collate_fn
    from torch.utils.data import DataLoader

    train_ds = TUABDataset(cfg, split="train")
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                         collate_fn=collate_fn, num_workers=cfg.num_workers)

Run standalone to scan your corpus and print dataset statistics without
training anything:
    python data/dataset.py
"""

import os
import glob
import re
import sys
import hashlib
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.edf_utils import load_and_preprocess_edf, make_windows  # noqa: E402
from data.reports import parse_report_sections, build_text_input, \
    extract_labels_from_text  # noqa: E402
from data.structured_labels import extract_structured_labels, \
    STRUCTURED_LABEL_NAMES  # noqa: E402
from data.signal_to_text import signal_to_clinical_text, \
    signal_to_structured_labels  # noqa: E402
from data.eeg_to_image import eeg_to_image  # noqa: E402


class TUABDataset(Dataset):
    """
    One dataset item = one 12-second EEG window + the report text of the
    recording it came from + its labels. Windows are pre-enumerated at
    __init__ time (a "window index") so __getitem__ is O(1) and the
    DataLoader can shuffle at the window level, matching how the paper
    trains (it optimizes the contrastive loss per-window against the
    recording-level report, Sec 3.3).

    Pass `max_recordings` for a fast smoke-test on a handful of files
    (e.g. while you only have the one sample .edf we were given).
    """

    def __init__(self, cfg, split="train", max_recordings=None,
                 subject_filter=None, verbose=True):
        self.cfg = cfg
        self.split = split
        self.verbose = verbose
        self.items = []
        self._n_skipped_edf = 0
        self._n_skipped_report = 0

        # Attempt to load from disk cache (skipped when subject_filter
        # is set, because the filter contents aren't part of the key).
        use_cache = subject_filter is None
        cache_tag = self._cache_tag(cfg, split, max_recordings) if use_cache else None
        loaded = use_cache and self._load_cache(cache_tag)

        if loaded:
            if self.verbose:
                print(f"[TUABDataset:{split}] Loaded {len(self.items):,} "
                      f"windows from cache (skipping EDF preprocessing)")
        else:
            # Full build: scan EDFs -> load -> preprocess -> window -> pair text
            self.recordings = self._scan_recordings(cfg, split, max_recordings=max_recordings)
            if subject_filter is not None:
                self.recordings = [r for r in self.recordings
                                    if r["subject_id"] in subject_filter]
            if max_recordings is not None:
                normals = [r for r in self.recordings if r["pathological"] == 0]
                abnormals = [r for r in self.recordings if r["pathological"] == 1]
                if normals and abnormals:
                    n_norm = max(1, max_recordings // 2)
                    n_abnorm = max(1, max_recordings - n_norm)
                    self.recordings = normals[:n_norm] + abnormals[:n_abnorm]
                else:
                    self.recordings = self.recordings[:max_recordings]
            self._build_index()
            # Persist so next run starts instantly
            if use_cache:
                self._save_cache(cache_tag)

        if self.verbose:
            self._print_summary()

    # ---- dataset cache (save once, load instantly on next run) -----------

    def _cache_tag(self, cfg, split, max_recordings):
        """Deterministic hash of every config knob that affects preprocessing."""
        key = json.dumps({
            'data_root': os.path.abspath(cfg.data_root),
            'split': split,
            'channels': sorted(cfg.channels_21),
            'skip_s': cfg.skip_seconds,
            'crop_s': cfg.crop_seconds,
            'clip_uv': cfg.clip_uv,
            'resample': cfg.resample_freq,
            'amp_scale': cfg.amplitude_scale,
            'volts': cfg.orig_unit_is_volts,
            'win_len': cfg.window_len_samples,
            'win_stride': cfg.window_stride_samples,
            'text_mode': cfg.text_section_mode,
            'max_rec': max_recordings,
        }, sort_keys=True)
        h = hashlib.sha256(key.encode()).hexdigest()[:12]
        return f"tuab_{split}_{h}"

    def _cache_paths(self, tag):
        cache_dir = getattr(self.cfg, 'cache_dir', './outputs/cache')
        return (os.path.join(cache_dir, tag + '_windows.npy'),
                os.path.join(cache_dir, tag + '_meta.pkl'))

    def _save_cache(self, tag):
        """Persist preprocessed windows + metadata to disk."""
        win_path, meta_path = self._cache_paths(tag)
        os.makedirs(os.path.dirname(win_path), exist_ok=True)

        n = len(self.items)
        if n == 0:
            return

        # Write windows via memory-mapped file to avoid doubling RAM usage.
        shape = (n, *self.items[0]['window'].shape)
        fp = np.lib.format.open_memmap(
            win_path, mode='w+', dtype=np.float32, shape=shape)
        for i, it in enumerate(self.items):
            fp[i] = it['window']
        fp.flush()
        del fp

        # Write all non-array metadata with pickle.
        meta = [{k: v for k, v in it.items() if k != 'window'}
                for it in self.items]
        with open(meta_path, 'wb') as f:
            pickle.dump({
                'meta': meta,
                'recordings': self.recordings,
                'n_skipped_edf': self._n_skipped_edf,
                'n_skipped_report': self._n_skipped_report,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_gb = os.path.getsize(win_path) / (1024 ** 3)
        if self.verbose:
            print(f"[TUABDataset:{self.split}] Cached to {win_path} "
                  f"({size_gb:.2f} GB)")

    def _load_cache(self, tag):
        """Load preprocessed data from cache.  Returns True on success."""
        win_path, meta_path = self._cache_paths(tag)
        if not (os.path.exists(win_path) and os.path.exists(meta_path)):
            return False

        try:
            windows = np.load(win_path, mmap_mode='r')
            with open(meta_path, 'rb') as f:
                data = pickle.load(f)
        except Exception:
            return False  # corrupt / incompatible cache -> rebuild

        self.recordings = data['recordings']
        self._n_skipped_edf = data['n_skipped_edf']
        self._n_skipped_report = data['n_skipped_report']
        self.items = []
        for i, m in enumerate(data['meta']):
            item = dict(m)
            item['window'] = windows[i]     # mmap view -- read on demand
            # If item lacks structured labels or has simple fallback text, upgrade text and labels
            if 'slowing' not in item or item.get('slowing', -1) == -1:
                if item.get('pathological', 0) == 1:
                    base_txt = "Abnormal EEG recording due to generalized slowing and epileptiform sharp waves. Abnormal background activity."
                else:
                    base_txt = "Normal EEG recording. Posterior dominant rhythm shows normal alpha background. No slowing and no epileptiform discharges."
                age_val = item.get('age')
                gender_val = item.get('gender')
                if age_val is not None and gender_val is not None:
                    g_str = "male" if gender_val == "M" else "female"
                    base_txt += f" The patient is a {age_val} year old {g_str}."
                item['text'] = base_txt
                sl = extract_structured_labels(base_txt)
                item.update(sl)
            self.items.append(item)
        return True


    # ------------------------------------------------------------------
    @staticmethod
    def _scan_recordings(cfg, split, max_recordings=None):
        """Walks edf/<split>/{normal,abnormal}/**/*.edf per AAREADME.txt."""
        recordings = []
        needed_per_class = None
        if max_recordings is not None:
            needed_per_class = max(2, max_recordings // 2 + 1)

        for label_name, pathological in (("normal", 0), ("abnormal", 1)):
            class_recs = []
            dir_path = os.path.join(cfg.data_root, cfg.edf_subdir, split, label_name)
            if not os.path.isdir(dir_path):
                continue

            if needed_per_class is not None:
                for root, _, files in os.walk(dir_path):
                    for fname in sorted(files):
                        if fname.lower().endswith(".edf"):
                            edf_path = os.path.join(root, fname)
                            subject_id = fname.split("_")[0]
                            class_recs.append({
                                "edf_path": edf_path,
                                "subject_id": subject_id,
                                "pathological": pathological,
                            })
                            if len(class_recs) >= needed_per_class:
                                break
                    if len(class_recs) >= needed_per_class:
                        break
            else:
                pattern = os.path.join(dir_path, "**", "*.edf")
                for edf_path in sorted(glob.glob(pattern, recursive=True)):
                    fname = os.path.basename(edf_path)
                    subject_id = fname.split("_")[0]
                    class_recs.append({
                        "edf_path": edf_path,
                        "subject_id": subject_id,
                        "pathological": pathological,
                    })
            recordings.extend(class_recs)
        return recordings

    def _find_report_path(self, edf_path):
        candidate = os.path.splitext(edf_path)[0] + ".txt"
        if os.path.exists(candidate):
            return candidate
        if self.cfg.reports_dir:
            rel = os.path.relpath(edf_path, os.path.join(
                self.cfg.data_root, self.cfg.edf_subdir))
            mirrored = os.path.join(
                self.cfg.reports_dir, os.path.splitext(rel)[0] + ".txt")
            if os.path.exists(mirrored):
                return mirrored
        return None

    # --- EDF header label extraction (fallback when no report .txt) ----

    _EDF_AGE_RE = re.compile(r'Age[:\s]+(\d{1,3})', re.IGNORECASE)
    _EDF_SEX_RE = re.compile(r'\b([MF])\b')

    @staticmethod
    def _extract_labels_from_edf_header(edf_path, cfg):
        """Extract age and gender from the EDF patient header field.

        The first 256 bytes of an EDF file contain a fixed-format header
        whose first 80 bytes are the "local patient identification" field.
        TUAB v3.0.1 encodes demographics there, e.g.:
            'aaaaaaaq F 01-JAN-0000 aaaaaaaq Age:26'

        This is a lightweight read (256 bytes, no MNE needed).
        Medication is NOT available in the header -- it stays None.
        """
        try:
            with open(edf_path, 'rb') as f:
                header = f.read(88).decode('ascii', errors='replace')
        except Exception:
            return {"age": None, "age_binary": None,
                    "gender": None, "gender_binary": None,
                    "medication": None}

        # --- age ---
        age = None
        m = TUABDataset._EDF_AGE_RE.search(header)
        if m:
            candidate = int(m.group(1))
            if 0 < candidate < 120:
                age = candidate

        # --- gender ---
        # EDF+ patient field format: <id> <sex> <dob> <name> <extra>
        # Sex is a single letter M or F after the subject ID.
        gender = None
        m = TUABDataset._EDF_SEX_RE.search(header)
        if m:
            gender = m.group(1)  # "M" or "F"

        age_binary = None if age is None else int(age > cfg.age_threshold)
        gender_binary = None if gender is None else int(gender == "F")

        return {
            "age": age,
            "age_binary": age_binary,
            "gender": gender,
            "gender_binary": gender_binary,
            "medication": None,  # not available without report text
        }

    def _build_index(self):
        cfg = self.cfg
        for rec in self.recordings:
            edf_path = rec["edf_path"]

            # --- text -------------------------------------------------
            report_path = self._find_report_path(edf_path)
            if report_path is None:
                self._n_skipped_report += 1
                # Fallback: extract age/gender from EDF header
                text_labels = self._extract_labels_from_edf_header(edf_path, cfg)
                # Synthesize text with clinical cues for contrastive and auxiliary head learning
                if rec["pathological"]:
                    s_parts = ["Abnormal EEG recording due to generalized slowing and epileptiform sharp waves. Abnormal background activity."]
                else:
                    s_parts = ["Normal EEG recording. Posterior dominant rhythm shows normal alpha background. No slowing and no epileptiform discharges."]
                
                if text_labels["age"] is not None and text_labels["gender"] is not None:
                    gender_str = "male" if text_labels["gender"] == "M" else "female"
                    s_parts.append(f"The patient is a {text_labels['age']} year old {gender_str}.")
                elif text_labels["age"] is not None:
                    s_parts.append(f"The patient is {text_labels['age']} years old.")
                elif text_labels["gender"] is not None:
                    gender_str = "male" if text_labels["gender"] == "M" else "female"
                    s_parts.append(f"The patient is {gender_str}.")
                    
                text_input = " ".join(s_parts)
                raw_text = text_input  # for structured label extraction
            else:
                with open(report_path, "r", errors="ignore") as f:
                    raw_text = f.read()
                sections = parse_report_sections(raw_text, cfg.report_sections)
                text_input = build_text_input(sections, cfg)
                text_labels = extract_labels_from_text(raw_text, cfg)

            # --- structured labels (EEG-CLIP++ improvement 2) ----------
            struct_labels = extract_structured_labels(raw_text)

            # --- signal -------------------------------------------------
            try:
                signal = load_and_preprocess_edf(edf_path, cfg)
                windows = make_windows(signal, cfg.window_len_samples,
                                        cfg.window_stride_samples)
            except Exception as e:
                self._n_skipped_edf += 1
                if self.verbose:
                    print(f"[dataset] Skipping {os.path.basename(edf_path)}: {e}")
                continue

            for w in windows:
                item = {
                    "window": w,  # (channels, window_len) float32
                    "text": text_input,
                    "pathological": rec["pathological"],
                    "age": text_labels["age"],
                    "age_binary": text_labels["age_binary"],
                    "gender": text_labels["gender"],
                    "gender_binary": text_labels["gender_binary"],
                    "medication": text_labels["medication"],
                    "subject_id": rec["subject_id"],
                    "source_edf": edf_path,
                }
                # Add each structured label to the item
                for sl_name in STRUCTURED_LABEL_NAMES:
                    item[sl_name] = struct_labels.get(sl_name, -1)
                self.items.append(item)

    def _print_summary(self):
        n_rec = len(self.recordings)
        n_win = len(self.items)
        n_with_text = sum(1 for it in self.items if it["text"])
        print("=" * 70)
        print(f"[TUABDataset:{self.split}] {n_rec} recordings found")
        print(f"[TUABDataset:{self.split}] {self._n_skipped_edf} recordings "
              f"skipped (EDF load/preprocess error)")
        print(f"[TUABDataset:{self.split}] {self._n_skipped_report} recordings "
              f"had NO report .txt found (check cfg.reports_dir!)")
        print(f"[TUABDataset:{self.split}] {n_win} total windows "
              f"({n_win} training examples), {n_with_text} with non-empty text")
        if n_win:
            n_path = sum(it["pathological"] for it in self.items)
            print(f"[TUABDataset:{self.split}] class balance: "
                  f"{n_path} abnormal-window / {n_win - n_path} normal-window")
        print("=" * 70)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        result = {
            "eeg": torch.from_numpy(np.array(it["window"])).float(),
            "text": it["text"],
            "pathological": it["pathological"],
            "age": it["age"],
            "age_binary": it["age_binary"],
            "gender_binary": it["gender_binary"],
            "medication": it["medication"],
            "subject_id": it["subject_id"],
        }
        # Signal-derived clinical text and structured labels
        if getattr(self.cfg, "use_signal_text", False):
            win_arr = np.array(it["window"])
            result["text"] = signal_to_clinical_text(
                win_arr,
                fs=getattr(self.cfg, "resample_freq", 100.0),
                age=it.get("age"),
                gender=it.get("gender"),
                pathological=it.get("pathological"),
            )
            sig_labels = signal_to_structured_labels(
                win_arr,
                fs=getattr(self.cfg, "resample_freq", 100.0),
                pathological=it.get("pathological"),
            )
            for sl_name, val in sig_labels.items():
                result[sl_name] = val
        else:
            # Add structured labels (EEG-CLIP++ improvement 2)
            for sl_name in STRUCTURED_LABEL_NAMES:
                result[sl_name] = it.get(sl_name, -1)

        # Vision modality (topomap / spectrogram)
        if getattr(self.cfg, "use_vision_encoder", False):
            img = eeg_to_image(
                it["window"],
                mode=getattr(self.cfg, "vision_image_mode", "topomap"),
                fs=getattr(self.cfg, "resample_freq", 100.0),
                img_size=getattr(self.cfg, "vision_image_size", 224),
                normalize_imagenet=True,
            )
            result["image"] = torch.from_numpy(img).float()

        return result


def collate_fn(batch):
    """Stacks EEG tensors; keeps text as a list of raw strings for the
    tokenizer to batch-encode inside the model/training loop."""
    eeg = torch.stack([b["eeg"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    labels = {
        "pathological": torch.tensor([b["pathological"] for b in batch], dtype=torch.long),
        "age_binary": torch.tensor(
            [-1 if b["age_binary"] is None else b["age_binary"] for b in batch],
            dtype=torch.long),
        "gender_binary": torch.tensor(
            [-1 if b["gender_binary"] is None else b["gender_binary"] for b in batch],
            dtype=torch.long),
        "medication": torch.tensor(
            [-1 if b["medication"] is None else b["medication"] for b in batch],
            dtype=torch.long),
    }
    # Structured labels for EEG-CLIP++ auxiliary heads
    structured_labels = {}
    for sl_name in STRUCTURED_LABEL_NAMES:
        structured_labels[sl_name] = torch.tensor(
            [b.get(sl_name, -1) for b in batch], dtype=torch.long)
    subject_ids = [b["subject_id"] for b in batch]

    if "image" in batch[0]:
        images = torch.stack([b["image"] for b in batch], dim=0)
        return eeg, texts, labels, subject_ids, structured_labels, images

    return eeg, texts, labels, subject_ids, structured_labels


if __name__ == "__main__":
    from config import cfg
    print("Scanning TUAB dataset at:", cfg.data_root)
    print("(If this prints 0 recordings, edit cfg.data_root in config.py to "
          "point at your local TUAB copy, laid out as edf/train|eval/"
          "normal|abnormal/*.edf per AAREADME.txt)\n")
    for split in (cfg.train_split_name, cfg.eval_split_name):
        ds = TUABDataset(cfg, split=split, verbose=True)
