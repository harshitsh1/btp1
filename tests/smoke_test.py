"""
tests/smoke_test.py
====================
WHY
---
A fast, no-GPU-required, no-full-dataset-required check that every stage
of the pipeline (EDF preprocessing -> windowing -> report parsing -> EEG
encoder forward pass -> contrastive loss) runs and produces the right
shapes, using ONLY the single sample .edf you were given. This is what
you should run first, before pointing train.py at the full TUAB corpus,
to catch environment/setup problems early.

HOW
---
    python tests/smoke_test.py /path/to/aaaaabdo_s003_t000.edf
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import cfg
from data.edf_utils import load_and_preprocess_edf, make_windows
from data.reports import parse_report_sections, build_text_input, extract_labels_from_text
from models.eeg_clip import EEGCLIP, clip_contrastive_loss


def main(edf_path):
    print("=" * 70)
    print("SMOKE TEST: EEG-CLIP pipeline on a single sample recording")
    print("=" * 70)

    print("\n[1/4] EDF preprocessing...")
    signal = load_and_preprocess_edf(edf_path, cfg, verbose=True)
    windows = make_windows(signal, cfg.window_len_samples, cfg.window_stride_samples)
    print(f"  -> {windows.shape[0]} windows of shape {windows.shape[1:]}")

    print("\n[2/4] Report parsing (using a synthetic report, since none was "
          "provided for this sample file)...")
    fake_report = (
        "CLINICAL HISTORY:\n55 year old male with history of epilepsy, on keppra.\n"
        "IMPRESSION:\nAbnormal EEG due to intermittent slowing.\n"
    )
    sections = parse_report_sections(fake_report, cfg.report_sections)
    text = build_text_input(sections, cfg)
    labels = extract_labels_from_text(fake_report, cfg)
    print(f"  -> text: {text!r}")
    print(f"  -> labels: {labels}")

    print("\n[3/4] Building EEG-CLIP model (dummy text encoder, CPU, for speed)...")
    import torch.nn as nn

    class _DummyTextEncoder(nn.Module):
        hidden_size = 32

        def __init__(self):
            super().__init__()
            self.embed = nn.EmbeddingBag(2000, self.hidden_size)

        def forward(self, texts, device=None):
            device = device or torch.device("cpu")
            idxs = [torch.randint(0, 2000, (8,)) for _ in texts]
            flat = torch.cat(idxs).to(device)
            offsets = torch.tensor(
                [0] + [len(i) for i in idxs[:-1]], device=device).cumsum(0)
            return self.embed(flat, offsets)

    model = EEGCLIP(cfg, text_encoder=_DummyTextEncoder())

    print("\n[4/4] Forward pass + contrastive loss on the real EEG windows...")
    eeg_batch = torch.from_numpy(windows[:min(8, len(windows))]).float()
    texts = [text] * eeg_batch.shape[0]
    eeg_z, text_z, scale = model(eeg_batch, texts)
    loss = clip_contrastive_loss(eeg_z, text_z, scale)
    print(f"  -> eeg_z shape: {tuple(eeg_z.shape)}")
    print(f"  -> text_z shape: {tuple(text_z.shape)}")
    print(f"  -> contrastive loss (untrained, sanity value): {loss.item():.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED. The pipeline runs end to end.")
    print("Next: point config.py's data_root at your full TUAB corpus and")
    print("run `python train.py`.")
    print("=" * 70)


if __name__ == "__main__":
    import glob
    path = sys.argv[1] if len(sys.argv) > 1 else None

    # If no path or path doesn't exist, auto-discover an .edf file from cfg.data_root
    if path is None or not os.path.exists(path):
        candidate = None
        data_root = os.path.abspath(cfg.data_root)
        if os.path.isdir(data_root):
            for root, _, files in os.walk(data_root):
                for f in files:
                    if f.lower().endswith(".edf"):
                        candidate = os.path.join(root, f)
                        break
                if candidate is not None:
                    break

        if candidate is not None:
            if path is not None and not os.path.exists(path):
                print(f"[smoke_test] Specified path '{path}' not found.")
            print(f"[smoke_test] Auto-discovered sample EDF: {candidate}\n")
            path = candidate
        else:
            print("Usage: python tests/smoke_test.py /path/to/sample.edf")
            print(f"Could not automatically locate an .edf file in cfg.data_root ({cfg.data_root}).")
            sys.exit(1)

    main(path)
