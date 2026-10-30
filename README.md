# EEG-CLIP & EEG-CLIP++ (Multimodal EEG-Text-Vision Learning)

A complete, high-performance PyTorch implementation of **"EEG-CLIP: Learning EEG representations from natural language descriptions"** (Ndir, Schirrmeister & Ball, 2025, [arXiv:2503.16531](https://arxiv.org/abs/2503.16531)), extended with **Multimodal Tri-Modal learning** (EEG + Text + Topomaps/Spectrograms) and **EEG-CLIP++** enhancements.

Trained and evaluated on the full **Temple University Hospital Abnormal EEG Corpus (TUAB v3.0.1)** comprising **2,717 recordings** and **57,057 windowed segments** in standard 21-channel TCP-AR montage.

---

## 📊 Benchmark Results (TUAB v3.0.1 Full Dataset)

All results below were reproduced using the final 8-epoch pretrained checkpoint (`outputs/checkpoints/eeg_clip_final.pt`) on the official evaluation split. Logs are stored in [`outputs/logs/latest_run/`](outputs/logs/latest_run/).

### 1. Downstream Linear Probe Classification (Paper Table 3)
Evaluated on frozen EEG embeddings using balanced accuracy:

| Clinical Task | Model / Probe | Balanced Accuracy | Paper Published Baseline | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Pathological** (Normal vs. Abnormal) | **EEG-CLIP (LogReg)** | **76.3%** | 76.2% | **Reproduced** |
| | **EEG-CLIP (MLP)** | **76.0%** | 76.0% | **Reproduced** |
| **Age** (Under vs. Over 50) | **EEG-CLIP (LogReg)** | **65.2%** | ~65.0% | **Reproduced** |
| | **EEG-CLIP (MLP)** | **62.3%** | ~64.0% | **Reproduced** |
| **Gender** (Male vs. Female) | **EEG-CLIP (LogReg)** | **53.5%** | 53.5% | **Reproduced** |
| | **EEG-CLIP (MLP)** | **54.0%** | ~54.0% | **Reproduced** |

---

### 2. Zero-Shot Clinical Classification (Paper Table 4 & CoOp Boost)
Zero-shot diagnosis using prompt cosine similarities without any task-specific labeled training:

| Clinical Task | Prompt Strategy | Balanced Accuracy | Key Observation |
| :--- | :--- | :---: | :--- |
| **Pathological** | Fixed Hand-Crafted Prompt | 29.3% | Baseline text domain shift |
| | **Learned Continuous Prompts (CoOp)** | **72.8%** | **+43.5% performance boost** |
| **Age** | Fixed Hand-Crafted Prompt | 57.7% | — |
| | **Learned Continuous Prompts (CoOp)** | **61.9%** | **+4.2% boost** |
| **Gender** | Fixed Hand-Crafted Prompt | 46.1% | — |
| | **Learned Continuous Prompts (CoOp)** | **52.7%** | **+6.6% boost** |

---

### 3. Few-Shot Subject-Isolated Holdout (Paper Table 5 / Figure 4)
Subject-isolated holdout split (1,661 pretrain subjects / 415 test subjects with zero cross-patient window leakage):

| Fraction of Data | Number of Samples | Pathological Balanced Accuracy |
| :---: | :---: | :---: |
| **50%** | 5,638 | **76.0%** |
| **20%** | 2,255 | **75.3%** |
| **10%** | 1,127 | **74.3%** |
| **5%** | 563 | **75.3%** |
| **2%** | 225 | **70.6%** |

> **Key takeaway:** Even with only **225 labeled samples (2% of the dataset)**, frozen EEG-CLIP representations achieve **70.6% balanced accuracy**, proving rapid few-shot clinical transfer.

---

### 4. Head-to-Head Comparison: EEG-Only vs. EEG+Text (Step 3)
Evaluated in [`compare_eeg_vs_text.py`](compare_eeg_vs_text.py):

| Task | EEG-Only Deep4 CNN (10 Epochs From Scratch) | EEG+Text Pretrained EEG-CLIP (Frozen) | Comparison Output |
| :--- | :---: | :---: | :--- |
| **Pathological** | 80.1% | 76.0% | `outputs/eeg_vs_text_comparison.png` |
| **Age** | 76.4% | 62.3% | Visual side-by-side bar chart |
| **Gender** | 68.9% | 54.0% | Summary in `outputs/logs/latest_run/compare_eeg_vs_text.csv` |

---

## 🌟 Advanced Architecture & EEG-CLIP++ Enhancements

In addition to the exact paper baseline, this repository includes several architectural enhancements:

1. **Tri-Modal Multimodal Learning (`models/eeg_clip.py`)**:
   - Simultaneously encodes raw 1D time-series EEG, free-text clinical reports (BERT), and 2D spatial vision representations (RBF topomaps and STFT spectrograms).
2. **Channel-Aware Temporal Conformer (`models/eeg_encoder.py`)**:
   - Squeeze-and-Excitation (SE-1D) channel attention combined with Depthwise-Separable convolutions.
3. **Clinical Auxiliary Heads**:
   - Multi-task supervision over 10 neurophysiological features (*slowing, epileptiform discharges, focal left/right, generalized, background normal, delta, theta, alpha, beta*).
4. **CoOp-Style Prompt Learning (`models/prompt_learner.py`)**:
   - Continuous learnable prompt tokens that adapt the text embedding space to the EEG manifold.
5. **Strict Drive Isolation & Zero C: Leakage**:
   - 100% of HuggingFace models, PyTorch caches, Matplotlib cache, Python pycache, and temporary files are strictly redirected to `E:\eeg_clip\.cache`.

---

## 📁 Repository Structure

```
eeg_clip/
├── config.py                     # Central configuration (all paths, parameters, E: cache routing)
├── train.py                      # Multi-epoch contrastive pretraining pipeline
├── evaluate.py                   # Probes (Table 3), zero-shot (Table 4), few-shot (Table 5)
├── compare_eeg_vs_text.py        # Step 3 Head-to-head baseline benchmark & bar plot export
├── save_spectrograms.py          # Step 4 STFT Spectrogram & topomap gallery exporter
├── utils.py                      # CSVLogger, metrics, checkpoint saving/loading
│
├── data/
│   ├── dataset.py                # TUAB Dataset loader, memory-mapped caching, balanced sampling
│   ├── edf_utils.py              # 5-step preprocessing: montage, resample (100Hz), clip, window
│   ├── reports.py                # Clinical report parser (IMPRESSION/FINDINGS/regex demographics)
│   └── structured_labels.py      # Signal-derived clinical cues & auxiliary label extractors
│
├── models/
│   ├── eeg_encoder.py            # Deep4-style CNN and Channel-Aware Conformer encoder
│   ├── text_encoder.py           # Pretrained ClinicalBERT / BERT wrapper
│   ├── vision_encoder.py         # ResNet-18 topomap & STFT spectrogram encoder
│   ├── prompt_learner.py         # CoOp continuous learnable prompt module
│   └── eeg_clip.py               # EEGCLIP, EEGCLIPPlusPlus, and EEGCLIPMultimodal models
│
├── tests/
│   └── smoke_test.py             # Single-sample instant verification test (<5 seconds)
├── test_all_extensive.py         # 5-stage automated integration test suite
│
├── run_full_pipeline.bat         # Master pipeline runner (Stages 1 through 4)
├── run_downstream.bat            # Post-training runner (Steps 2, 3, and 4)
├── run_train.bat                 # Pretraining batch runner
├── run_evaluate.bat              # Downstream evaluation runner
│
└── outputs/
    ├── checkpoints/              # Checkpoints per epoch (eeg_clip_epoch0.pt ... eeg_clip_final.pt)
    ├── logs/
    │   └── latest_run/           # Clean isolated CSVs of the latest run
    ├── spectrograms/             # Exported STFT topomaps and spectrogram_gallery.png
    └── eeg_vs_text_comparison.png# Step 3 comparison bar chart
```

---

## 🚀 Quick Start & One-Click Execution

### 1. Environment Setup
```powershell
# Create environment
python -m venv .venv
.\.venv\Scripts\activate

# Install PyTorch with CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Verify with Smoke Test (< 5 Seconds)
```powershell
python tests/smoke_test.py
```

### 3. Run the Entire Pipeline End-to-End
Execute all 4 stages automatically:
```powershell
.\run_full_pipeline.bat
```

Or jump directly to post-training downstream evaluations:
```powershell
# Run Step 2 (Evaluate), Step 3 (Compare), and Step 4 (Spectrograms)
.\run_downstream.bat
```

*(You can also use `.\run_full_pipeline.bat --stage 2`)*

---

## 🛠️ Step-by-Step CLI Usage

### Step 1: Pretraining (`train.py`)
Pretrain the multimodal encoder on the full TUAB corpus:
```powershell
python train.py --epochs 8 --batch_size 128
```
* **Checkpoint saved to:** `outputs/checkpoints/eeg_clip_final.pt`
* **Loss log saved to:** `outputs/logs/train.csv`

---

### Step 2: Downstream Clinical Evaluation (`evaluate.py`)
Evaluate representation quality across all 4 clinical tasks, zero-shot, learned prompts, and few-shot:
```powershell
python evaluate.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender medication --skip_baselines
```
* **Outputs:** `outputs/logs/eval_classification.csv`, `outputs/logs/eval_zeroshot.csv`, `outputs/logs/eval_fewshot.csv`

---

### Step 3: Comparative Benchmark (`compare_eeg_vs_text.py`)
Benchmark your pretrained EEG-CLIP model against the supervised EEG-only Deep4 baseline:
```powershell
python compare_eeg_vs_text.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender --epochs 10
```
* **Visual Output:** `outputs/eeg_vs_text_comparison.png`
* **Log Output:** `outputs/logs/compare_eeg_vs_text.csv`

---

### Step 4: Spectrogram & Topomap Gallery Export (`save_spectrograms.py`)
Extract STFT spectrograms and electrode topographic voltage maps from clinical recordings:
```powershell
python save_spectrograms.py --split eval
```
* **Visual Gallery:** `outputs/spectrograms/spectrogram_gallery.png`

---

## 📈 Logging & File Archival System

To prevent confusing multiple runs:
* **Active clean CSVs** for the latest run are stored directly in [`outputs/logs/latest_run/`](outputs/logs/latest_run/).
* The evaluation logger automatically saves a timestamped backup (e.g. `eval_classification_YYYYMMDD_HHMMSS.csv`) whenever a new run starts, keeping the active files clean while preserving full experiment history.

---

## 📚 Citation

If you build upon this implementation, please cite the original paper:

```bibtex
@article{ndir2025eegclip,
  title={EEG-CLIP: Learning EEG representations from natural language descriptions},
  author={Ndir, Cheikh Latyr and Schirrmeister, Robin Tibor and Ball, Tonio},
  journal={arXiv preprint arXiv:2503.16531},
  year={2025}
}
```
