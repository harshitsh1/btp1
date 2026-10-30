"""
config.py
=========
WHY
---
Every hyperparameter used anywhere in this codebase is declared here, in one
place, so that:
  1. You can reproduce the exact settings reported in the EEG-CLIP paper
     (Ndir, Schirrmeister & Ball, 2025 - arXiv:2503.16531) without hunting
     through every file.
  2. You can override a single value (e.g. switch text encoder, or run on a
     tiny debug subset) without touching model/training code.

HOW
---
Import the `Config` dataclass wherever you need a setting:
    from config import cfg
    print(cfg.batch_size)

Run this file directly to sanity-print the full configuration:
    python config.py
"""

from dataclasses import dataclass, field
from typing import List
import os
import sys

# Prevent pyarrow from loading; pyarrow 24.0.0 on Windows/Python 3.12 has a known
# static destructor crash on process exit (0xc0000005) that dumps 300MB files into AppData\Local\CrashDumps.
class _BlockPyarrowFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError(f"No module named '{fullname}' (blocked to prevent crash on exit)")
        return None

if not any(isinstance(f, _BlockPyarrowFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockPyarrowFinder())

# Force all cache, huggingface, torch, cuda, and temp storage to E: drive to prevent C: drive exhaustion
E_CACHE = "E:/eeg_clip/.cache"
os.environ["TORCH_HOME"] = f"{E_CACHE}/torch"
os.environ["HF_HOME"] = f"{E_CACHE}/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{E_CACHE}/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{E_CACHE}/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = f"{E_CACHE}/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = f"{E_CACHE}/huggingface/metrics"
os.environ["HF_MODULES_CACHE"] = f"{E_CACHE}/huggingface/modules"
os.environ["MPLCONFIGDIR"] = f"{E_CACHE}/matplotlib"
os.environ["TEMP"] = f"{E_CACHE}/temp"
os.environ["TMP"] = f"{E_CACHE}/temp"
os.environ["TMPDIR"] = f"{E_CACHE}/temp"
os.environ["CUDA_CACHE_PATH"] = f"{E_CACHE}/nvidia"
os.environ["PYTHONPYCACHEPREFIX"] = f"{E_CACHE}/pycache"

for p in [f"{E_CACHE}/torch", f"{E_CACHE}/huggingface", f"{E_CACHE}/matplotlib",
          f"{E_CACHE}/temp", f"{E_CACHE}/nvidia", f"{E_CACHE}/pycache"]:
    os.makedirs(p, exist_ok=True)

import tempfile
tempfile.tempdir = f"{E_CACHE}/temp"



@dataclass
class Config:
    # ------------------------------------------------------------------
    # Paths -- EDIT THESE to point at your local copy of the TUH Abnormal
    # (TUAB) corpus, laid out exactly as described in AAREADME.txt:
    #   <root>/edf/train/normal/...   <root>/edf/train/abnormal/...
    #   <root>/edf/eval/normal/...    <root>/edf/eval/abnormal/...
    # ------------------------------------------------------------------
    data_root: str = os.environ.get("TUAB_ROOT", "../tuh_eeg_data/v3.0.1")
    edf_subdir: str = "edf"
    # Directory (or list of directories) that contains one .txt report per
    # recording. TUAB ships reports as free text; if your local copy stores
    # them next to the .edf files (same stem, .txt extension) leave this as
    # None. Otherwise point it at wherever the *_report.txt files live.
    reports_dir: str = None
    train_split_name: str = "train"
    eval_split_name: str = "eval"

    output_dir: str = "E:/eeg_clip/outputs"
    checkpoint_dir: str = "E:/eeg_clip/outputs/checkpoints"
    log_dir: str = "E:/eeg_clip/outputs/logs"
    cache_dir: str = "E:/eeg_clip/outputs/cache"    # preprocessed dataset cache

    # ------------------------------------------------------------------
    # EEG preprocessing (paper Sec. 3.2, following Schirrmeister et al. 2017)
    # ------------------------------------------------------------------
    # 21 electrodes present in every TUAB recording (10-20 system, AR
    # referential montage). These are the *bare* 10-20 names; the actual
    # channel-name matching (which strips "EEG "/"-REF"/"-LE" etc. prefixes
    # and suffixes used by TUH's EDF headers) happens in data/edf_utils.py.
    channels_21: List[str] = field(default_factory=lambda: [
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
    ])
    skip_seconds: float = 60.0          # exclude first minute
    crop_seconds: float = 120.0         # keep next 2 minutes
    clip_uv: float = 800.0              # clip amplitude to +/- 800 microvolt
    resample_freq: float = 100.0        # target sampling rate (Hz)
    amplitude_scale: float = 30.0       # divide by 30 to approach unit variance
    orig_unit_is_volts: bool = True     # EDF signals are read in Volts by MNE

    # Windowing (paper Sec. 3.3): 1200 samples @ 100Hz = 12s, stride 519
    # (stride chosen by the paper so that consecutive windows tile the whole
    # 2-minute crop without gaps).
    window_len_samples: int = 1200
    window_stride_samples: int = 519

    # ------------------------------------------------------------------
    # Text / report processing (paper Sec. 3.1 & Table 1)
    # ------------------------------------------------------------------
    report_sections: List[str] = field(default_factory=lambda: [
        "IMPRESSION", "DESCRIPTION OF THE RECORD", "CLINICAL HISTORY",
        "MEDICATIONS", "INTRODUCTION", "CLINICAL CORRELATION", "HEART RATE",
        "FINDINGS", "REASON FOR STUDY", "TECHNICAL DIFFICULTIES", "EVENTS",
        "CONDITION OF THE RECORDING", "PAST MEDICAL HISTORY", "TYPE OF STUDY",
        "ACTIVATION PROCEDURES",
    ])
    # Which section(s) to feed the text encoder with during training.
    # "all" concatenates every non-empty section (best overall performance,
    # Fig. 5 in the paper). Set to e.g. "IMPRESSION" to reproduce the
    # section-ablation study of Sec. 4.2.
    text_section_mode: str = "all"
    max_text_tokens: int = 256

    medication_keywords: List[str] = field(
        default_factory=lambda: ["keppra", "dilantin", "depakote"]
    )
    age_threshold: int = 50  # binary age split used for the "Age" task

    # ------------------------------------------------------------------
    # Model architecture (paper Sec. 3.3, Fig. 1)
    # ------------------------------------------------------------------
    n_channels: int = 21
    eeg_embed_dim: int = 128          # Deep4 output embedding size
    projection_dim: int = 64          # shared contrastive space (Sec. 4.3.2)
    text_encoder_name: str = "bert-base-uncased"
    # Other options studied in the paper (Fig. 6):
    #   "medicalai/ClinicalBERT"
    #   "microsoft/BioGPT-Large-PubMedQA"
    #   "BAAI/bge-large-en-v1.5"
    freeze_text_encoder_layers: int = 10  # 10 = freeze lower layers to fit comfortably in laptop GPU memory

    # ------------------------------------------------------------------
    # Training (paper Sec. 3.3)
    # ------------------------------------------------------------------
    batch_size: int = 64
    n_epochs: int = 20
    eeg_lr: float = 1e-3
    weight_decay: float = 1e-4
    # Optimal text-encoder / EEG-encoder LR ratio found in Fig. 7
    text_lr_ratio: float = 1e-3
    temperature: float = 0.07             # CLIP softmax temperature (tau)
    #num_workers: int = 4
    num_workers: int = 0  # 0 on Windows: data is pre-loaded in RAM, so
                           # multiprocessing workers only add pickle overhead
                           # (and cause MemoryError with spawn-based mp).
    seed: int = 42
    device: str = "cuda"  # falls back to "cpu" automatically if unavailable

    # ------------------------------------------------------------------
    # Evaluation (paper Sec. 3.4)
    # ------------------------------------------------------------------
    eval_tasks: List[str] = field(
        default_factory=lambda: ["pathological", "age", "gender", "medication"]
    )
    few_shot_holdout_frac: float = 0.20   # 20% of TRAIN held out for few-shot
    few_shot_fractions: List[float] = field(
        default_factory=lambda: [1/2, 1/5, 1/10, 1/20, 1/50]
    )
    zero_shot_prompts: dict = field(default_factory=lambda: {
        "pathological": ("This is a normal recording", "This is an abnormal recording"),
        "age": ("The patient is under 50 years old", "The patient is over 50 years old"),
        "gender": ("The patient is male", "The patient is female"),
        "medication": (
            "No anti-epileptic drugs were prescribed to the patient",
            "Anti-epileptic drugs were prescribed to the patient",
        ),
    })

    # ==================================================================
    # EEG-CLIP++ extensions (all gated: set use_*=True to enable)
    # ==================================================================

    # ------------------------------------------------------------------
    # Loss function (paper improvement 1: SigLIP / soft targets)
    # ------------------------------------------------------------------
    # Options: "infonce" (original CLIP), "sigmoid" (SigLIP), "soft_target"
    loss_type: str = "infonce"
    # For soft-target loss: temperature for softening text-similarity targets
    soft_target_temperature: float = 0.05

    # ------------------------------------------------------------------
    # Structured-finding auxiliary heads (improvement 2)
    # ------------------------------------------------------------------
    use_auxiliary_heads: bool = True
    # Weight of auxiliary loss relative to contrastive loss:
    #   L_total = L_contrastive + aux_loss_weight * L_auxiliary
    aux_loss_weight: float = 0.5
    # Which structured labels to use as auxiliary targets
    structured_label_names: List[str] = field(default_factory=lambda: [
        "slowing", "epileptiform", "focal_left", "focal_right",
        "generalized", "background_normal",
        "delta", "theta", "alpha", "beta",
    ])
    # "rule" (regex, fast, default) or "llm" (requires OpenAI API key)
    structured_label_extractor: str = "rule"
    aux_hidden_dim: int = 64  # hidden dim for auxiliary head MLPs

    # ------------------------------------------------------------------
    # Channel-aware encoder (improvement 3)
    # ------------------------------------------------------------------
    use_channel_aware_encoder: bool = False
    channel_encoder_d_model: int = 128
    channel_encoder_n_layers: int = 2
    channel_encoder_n_heads: int = 4
    use_window_pooling: bool = False  # attention pooling over windows
    # NOTE: window pooling changes the training loop (batch = recordings
    # not windows). Off by default because it requires grouped batching.

    # ------------------------------------------------------------------
    # Learned prompts for zero-shot (improvement 4: CoOp-style)
    # ------------------------------------------------------------------
    use_learned_prompts: bool = True
    n_prompt_tokens: int = 8
    prompt_class_specific: bool = False
    # Template to initialize from (or "random")
    prompt_init_template: str = "random"
    prompt_lr: float = 1e-3  # LR for prompt token optimization
    # Class names per task for prompt learning
    prompt_class_names: dict = field(default_factory=lambda: {
        "pathological": ["normal", "abnormal"],
        "age":          ["young patient under 50", "elderly patient over 50"],
        "gender":       ["male patient", "female patient"],
        "medication":   ["no anti-epileptic medication",
                         "prescribed anti-epileptic medication"],
    })

    # ------------------------------------------------------------------
    # Multimodal EEG-CLIP extensions (Signal-to-Text & Vision Branch)
    # ------------------------------------------------------------------
    use_signal_text: bool = True           # derive individualized clinical text from EEG signal features
    use_vision_encoder: bool = False       # enable 3rd modality (Vision/Topomap/Spectrogram)
    vision_encoder_name: str = "resnet18"  # "resnet18" (fast/lightweight) or "vit_b_16" or "resnet50"
    vision_freeze_backbone: bool = True    # freeze pretrained visual backbone
    vision_image_mode: str = "topomap"     # "topomap", "spectrogram", or "hybrid"
    vision_image_size: int = 224           # 224x224 RGB image
    vision_lr_ratio: float = 1e-3          # LR ratio for vision projection head relative to EEG LR
    vision_loss_weight: float = 1.0        # weight for L_eeg<->vision InfoNCE loss
    text_vision_loss_weight: float = 0.5   # weight for L_text<->vision InfoNCE loss


cfg = Config()


if __name__ == "__main__":
    import json
    from dataclasses import asdict
    print("=" * 70)
    print("EEG-CLIP CONFIGURATION")
    print("=" * 70)
    for k, v in asdict(cfg).items():
        print(f"  {k:28s}: {v}")
    print("=" * 70)
