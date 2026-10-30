"""
data/edf_utils.py
==================
WHY
---
The EEG-CLIP paper (Sec. 3.2) preprocesses every recording with five fixed
steps, copied from Schirrmeister et al. (2017):
    1. Keep a fixed 21-electrode subset present in *every* recording.
    2. Drop the first minute (often noisy / electrode settling) and keep
       only the next 2 minutes.
    3. Clip amplitudes to +/-800 microvolt (kills big movement artifacts).
    4. Resample to 100 Hz (cuts compute cost, still keeps clinically
       relevant <=40Hz content).
    5. Divide by 30 (crude variance normalization so the CNN doesn't have
       to learn amplitude scale from scratch).

This file implements exactly those five steps on top of MNE-Python, which
handles the many EDF header quirks (channel naming, reference schemes,
units) found in real clinical recordings such as the one you gave us
(aaaaabdo_s003_t000.edf: 30 raw channels, 250 Hz, TCP-AR montage with
"EEG <NAME>-REF" channel naming).

HOW
---
    from data.edf_utils import load_and_preprocess_edf
    x = load_and_preprocess_edf("/path/to/file.edf", cfg)
    # x: np.ndarray, shape (21, n_samples_after_resample)

Run this file directly on a sample EDF to see every step print its effect:
    python data/edf_utils.py /path/to/sample.edf
"""

import sys
import os
import numpy as np

try:
    import mne
    mne.set_log_level("ERROR")
except ImportError as e:
    raise ImportError(
        "mne is required for EDF I/O. Install with `pip install mne`."
    ) from e


def _canonical(name: str) -> str:
    """
    WHY: TUH EDF headers name channels like 'EEG FP1-REF' or 'EEG A1-LE'.
    We strip the 'EEG ' prefix and the '-REF'/'-LE'/'-AR' reference suffix
    so we can match against our canonical 10-20 names (e.g. 'FP1').
    """
    n = name.upper().strip()
    n = n.replace("EEG ", "")
    for suffix in ("-REF", "-LE", "-AR", "-AVG"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()


def select_21_channels(raw: "mne.io.BaseRaw", channels_21):
    """
    Reorders/selects the fixed 21-channel subset. Raises a clear error
    (rather than silently zero-padding) if a recording is missing a
    channel, since the paper explicitly says the 21 channels are "present
    in all recordings" -- a missing channel signals a corrupted/atypical
    file that should be skipped by the caller.
    """
    name_map = {_canonical(ch): ch for ch in raw.ch_names}
    missing = [c for c in channels_21 if c not in name_map]
    if missing:
        raise ValueError(
            f"Recording is missing required channels: {missing}. "
            f"Available channels: {raw.ch_names}"
        )
    ordered_raw_names = [name_map[c] for c in channels_21]
    raw = raw.copy().pick(ordered_raw_names)
    raw.reorder_channels(ordered_raw_names)
    return raw


def load_and_preprocess_edf(path: str, cfg, verbose: bool = False) -> np.ndarray:
    """
    Implements paper Sec 3.2 preprocessing end to end.

    Returns
    -------
    np.ndarray of shape (n_channels, n_samples) at cfg.resample_freq Hz,
    already clipped and amplitude-scaled, ready for windowing.
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

    if verbose:
        print(f"[edf_utils] Loaded {os.path.basename(path)}: "
              f"{len(raw.ch_names)} channels, {raw.info['sfreq']} Hz, "
              f"{raw.n_times / raw.info['sfreq']:.1f}s")

    # Step 1: fixed 21-channel subset -----------------------------------
    raw = select_21_channels(raw, cfg.channels_21)
    if verbose:
        print(f"[edf_utils] Selected {len(cfg.channels_21)} channels: "
              f"{cfg.channels_21}")

    sfreq = raw.info["sfreq"]
    total_seconds = raw.n_times / sfreq
    needed = cfg.skip_seconds + cfg.crop_seconds
    if total_seconds < needed:
        raise ValueError(
            f"Recording too short ({total_seconds:.1f}s) to skip "
            f"{cfg.skip_seconds}s and crop {cfg.crop_seconds}s."
        )

    # Step 2: drop first minute, keep next 2 minutes ---------------------
    raw.crop(tmin=cfg.skip_seconds, tmax=cfg.skip_seconds + cfg.crop_seconds,
              include_tmax=False)
    if verbose:
        print(f"[edf_utils] Cropped to [{cfg.skip_seconds}, "
              f"{cfg.skip_seconds + cfg.crop_seconds})s "
              f"-> {raw.n_times} samples @ {sfreq}Hz")

    data = raw.get_data()  # Volts, shape (channels, samples)
    if cfg.orig_unit_is_volts:
        data = data * 1e6  # convert to microvolts to match +/-800uV clip

    # Step 3: clip to +/-800 uV ------------------------------------------
    n_clipped = int(np.sum(np.abs(data) > cfg.clip_uv))
    data = np.clip(data, -cfg.clip_uv, cfg.clip_uv)
    if verbose:
        pct = 100.0 * n_clipped / data.size
        print(f"[edf_utils] Clipped {n_clipped} samples "
              f"({pct:.3f}% of all samples) to +/-{cfg.clip_uv}uV")

    # Step 4: resample to target frequency --------------------------------
    if sfreq != cfg.resample_freq:
        n_new = int(round(data.shape[1] * cfg.resample_freq / sfreq))
        data = mne.filter.resample(data, up=cfg.resample_freq, down=sfreq,
                                    npad="auto")
        if verbose:
            print(f"[edf_utils] Resampled {sfreq}Hz -> {cfg.resample_freq}Hz "
                  f"({data.shape[1]} samples)")

    # Step 5: divide by 30 -------------------------------------------------
    data = data / cfg.amplitude_scale
    if verbose:
        print(f"[edf_utils] Scaled by 1/{cfg.amplitude_scale}. "
              f"Final shape: {data.shape}, "
              f"mean={data.mean():.4f}, std={data.std():.4f}")

    return data.astype(np.float32)


def make_windows(data: np.ndarray, window_len: int, stride: int) -> np.ndarray:
    """
    Slices a (channels, samples) array into overlapping windows.

    WHY: the paper feeds the Deep4 encoder fixed-length windows (1200
    samples = 12s @ 100Hz) with stride 519, "which ensures all timesteps
    are predicted without any gap by our Deep4 model" (Sec 3.3) -- i.e.
    receptive-field-aware tiling of the 2-minute crop.

    Returns
    -------
    np.ndarray of shape (n_windows, channels, window_len)
    """
    n_channels, n_samples = data.shape
    starts = list(range(0, n_samples - window_len + 1, stride))
    if not starts:
        raise ValueError(
            f"Recording of {n_samples} samples is shorter than one window "
            f"of {window_len} samples."
        )
    windows = np.stack([data[:, s:s + window_len] for s in starts], axis=0)
    return windows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import cfg

    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        print("Usage: python data/edf_utils.py /path/to/file.edf")
        sys.exit(1)

    print("=" * 70)
    print(f"Preprocessing demo on: {path}")
    print("=" * 70)
    x = load_and_preprocess_edf(path, cfg, verbose=True)
    w = make_windows(x, cfg.window_len_samples, cfg.window_stride_samples)
    print(f"[edf_utils] Produced {w.shape[0]} windows of shape "
          f"{w.shape[1:]} (channels x samples)")
