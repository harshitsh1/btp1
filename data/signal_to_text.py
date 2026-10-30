"""
data/signal_to_text.py
======================
Signal-to-Text Synthesis for EEG-CLIP.

WHY
---
When clinical free-text reports (.txt) are missing or sparse (e.g. in TUAB
without local reports), standard training relies on repetitive hardcoded strings.
This module extracts quantitative EEG (qEEG) features directly from the 21-channel
12-second EEG window:
  1. Power Spectral Density (PSD) for Delta (0.5-4 Hz), Theta (4-8 Hz),
     Alpha (8-13 Hz), and Beta (14-30 Hz).
  2. Regional groupings: Frontal, Central, Temporal, Parietal, Occipital, Left vs Right.
  3. Posterior Dominant Rhythm (PDR) organization and peak frequency.
  4. Diffuse slowing index = (Delta + Theta) / (Alpha + Beta).
  5. Hemispheric and focal regional asymmetry (e.g. Left vs Right Temporal).
  6. Transient sharpness / kurtosis proxy for epileptiform discharge likelihood.

It transforms these mathematical features into an individualized, clinical-grade
neurophysiological narrative text for the BERT Text Encoder to align with.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

CHANNELS_21 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
]

CH_TO_IDX = {ch: i for i, ch in enumerate(CHANNELS_21)}

# Anatomical channel groups
REGIONS = {
    "frontal": ["FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"],
    "central": ["C3", "C4", "CZ"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "parietal": ["P3", "P4", "PZ"],
    "occipital": ["O1", "O2"],
    "left": ["FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "A1"],
    "right": ["FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6", "A2"],
    "left_temporal": ["T3", "T5"],
    "right_temporal": ["T4", "T6"],
    "midline": ["FZ", "CZ", "PZ"],
}

REGION_IDXS = {
    region: [CH_TO_IDX[ch] for ch in ch_list if ch in CH_TO_IDX]
    for region, ch_list in REGIONS.items()
}

FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (14.0, 30.0),
}


def compute_psd(
    signal: np.ndarray, fs: float = 100.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Power Spectral Density for (channels, samples) using real FFT.
    Returns (freqs, psd) where freqs is (n_freqs,) and psd is (channels, n_freqs).
    """
    n_samples = signal.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    # Hanning window to prevent spectral leakage
    window = np.hanning(n_samples).astype(signal.dtype)
    fft_vals = np.fft.rfft(signal * window, axis=-1)
    psd = (np.abs(fft_vals) ** 2) / (np.sum(window ** 2) * fs)
    # Double power for non-DC/non-Nyquist bins
    psd[..., 1:-1] *= 2.0
    return freqs, psd


def extract_qeeg_metrics(
    window: Union[np.ndarray, torch.Tensor], fs: float = 100.0
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Extracts quantitative EEG features from a 21-channel EEG window.
    Shape: (21, n_samples).
    """
    if isinstance(window, torch.Tensor):
        sig = window.detach().cpu().numpy()
    else:
        sig = np.asarray(window, dtype=np.float32)

    if sig.ndim == 1:
        sig = sig[np.newaxis, :]

    n_channels, n_samples = sig.shape
    freqs, psd = compute_psd(sig, fs=fs)

    # Total physiological power (0.5 to 30 Hz)
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = np.sum(psd[:, total_mask], axis=-1) + 1e-8  # (21,)

    # Band powers per channel
    band_powers_abs = {}
    band_powers_rel = {}
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        band_mask = (freqs >= f_low) & (freqs < f_high)
        bp_abs = np.sum(psd[:, band_mask], axis=-1)  # (21,)
        bp_rel = bp_abs / total_power  # (21,)
        band_powers_abs[band_name] = bp_abs
        band_powers_rel[band_name] = bp_rel

    # Global relative band powers (averaged across all channels)
    global_rel = {
        band: float(np.mean(band_powers_rel[band])) for band in FREQ_BANDS
    }

    # Regional relative band powers
    regional_rel = {}
    for region, idxs in REGION_IDXS.items():
        regional_rel[region] = {
            band: float(np.mean(band_powers_rel[band][idxs]))
            for band in FREQ_BANDS
        }

    # 1. Slowing Index: (Delta + Theta) / (Alpha + Beta)
    slow_p = global_rel["delta"] + global_rel["theta"]
    fast_p = global_rel["alpha"] + global_rel["beta"] + 1e-6
    slowing_index = float(slow_p / fast_p)

    # 2. Posterior Dominant Rhythm (PDR) assessment over occipital channels (O1, O2)
    occ_idxs = REGION_IDXS["occipital"]
    occ_alpha_rel = regional_rel["occipital"]["alpha"]
    occ_slow_rel = regional_rel["occipital"]["delta"] + regional_rel["occipital"]["theta"]
    
    # Peak alpha frequency in occipital channels
    alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
    occ_psd_alpha = np.mean(psd[occ_idxs][:, alpha_mask], axis=0)
    alpha_freqs = freqs[alpha_mask]
    if len(alpha_freqs) > 0 and occ_alpha_rel > 0.08:
        peak_alpha_freq = float(alpha_freqs[np.argmax(occ_psd_alpha)])
    else:
        peak_alpha_freq = 0.0

    # 3. Hemispheric Asymmetry Index: (Left - Right) / (Left + Right)
    left_idxs = REGION_IDXS["left"]
    right_idxs = REGION_IDXS["right"]
    left_tot = float(np.mean(total_power[left_idxs]))
    right_tot = float(np.mean(total_power[right_idxs]))
    hemispheric_asymmetry = float((left_tot - right_tot) / (left_tot + right_tot + 1e-6))

    # Left vs Right Temporal Slowing Asymmetry
    lt_idxs = REGION_IDXS["left_temporal"]
    rt_idxs = REGION_IDXS["right_temporal"]
    lt_slow = float(np.mean(band_powers_rel["delta"][lt_idxs] + band_powers_rel["theta"][lt_idxs]))
    rt_slow = float(np.mean(band_powers_rel["delta"][rt_idxs] + band_powers_rel["theta"][rt_idxs]))
    temporal_asymmetry = float((lt_slow - rt_slow) / (lt_slow + rt_slow + 1e-6))

    # 4. Epileptiform / sharp transient proxy (kurtosis and peak-to-RMS ratio)
    diff_sig = np.diff(sig, axis=-1)
    kurtosis_per_ch = np.mean(diff_sig ** 4, axis=-1) / (np.var(diff_sig, axis=-1) ** 2 + 1e-8)
    max_kurtosis = float(np.max(kurtosis_per_ch))

    return {
        "global_rel": global_rel,
        "regional_rel": regional_rel,
        "slowing_index": slowing_index,
        "pdr_alpha_rel": occ_alpha_rel,
        "pdr_slow_rel": occ_slow_rel,
        "peak_alpha_freq": peak_alpha_freq,
        "hemispheric_asymmetry": hemispheric_asymmetry,
        "temporal_asymmetry": temporal_asymmetry,
        "max_kurtosis": max_kurtosis,
    }


def signal_to_clinical_text(
    window: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    pathological: Optional[int] = None,
    detailed: bool = True,
) -> str:
    """
    Translates raw EEG signal features into a coherent, clinically realistic narrative.
    """
    metrics = extract_qeeg_metrics(window, fs=fs)
    g_rel = metrics["global_rel"]
    r_rel = metrics["regional_rel"]
    slow_idx = metrics["slowing_index"]
    pdr_alpha = metrics["pdr_alpha_rel"]
    peak_alpha = metrics["peak_alpha_freq"]
    hemi_asym = metrics["hemispheric_asymmetry"]
    temp_asym = metrics["temporal_asymmetry"]
    kurt = metrics["max_kurtosis"]

    sentences = []

    # 1. Background & Posterior Dominant Rhythm
    if pdr_alpha >= 0.22 and peak_alpha >= 8.0:
        sentences.append(
            f"Posterior dominant rhythm is well-organized at {peak_alpha:.1f} Hz "
            f"over occipital regions (O1, O2) with prominent alpha activity ({pdr_alpha * 100:.1f}%)."
        )
    elif pdr_alpha >= 0.12:
        sentences.append(
            f"Posterior background demonstrates moderate alpha rhythm at {peak_alpha:.1f} Hz "
            f"with some attenuation ({pdr_alpha * 100:.1f}% relative power)."
        )
    else:
        sentences.append(
            "Posterior dominant alpha rhythm is suppressed or poorly organized, "
            "with marked slowing replacing the normal posterior gradient."
        )

    # 2. Slow Wave Activity & Encephalopathy assessment
    delta_pct = g_rel["delta"] * 100.0
    theta_pct = g_rel["theta"] * 100.0
    if slow_idx >= 2.0 or (g_rel["delta"] + g_rel["theta"] >= 0.65):
        frontal_slow = r_rel["frontal"]["delta"] * 100.0
        sentences.append(
            f"Severe diffuse generalized slowing is present with prominent delta ({delta_pct:.1f}%) "
            f"and theta ({theta_pct:.1f}%) frequency activity (slowing index {slow_idx:.2f}), "
            f"most pronounced over frontal channels (FP1, FP2, F3, F4: {frontal_slow:.1f}% delta)."
        )
    elif slow_idx >= 1.2 or (g_rel["delta"] + g_rel["theta"] >= 0.50):
        sentences.append(
            f"Mild to moderate generalized theta slowing ({theta_pct:.1f}%) and delta activity "
            f"({delta_pct:.1f}%) is observed across anterior and central derivations."
        )
    else:
        sentences.append(
            f"Normal physiological background without excessive slow activity "
            f"(delta {delta_pct:.1f}%, theta {theta_pct:.1f}%, slowing index {slow_idx:.2f})."
        )

    # 3. Focal & Regional Asymmetries
    if abs(temp_asym) >= 0.25:
        affected_side = "left" if temp_asym > 0 else "right"
        channels = "T3, T5" if temp_asym > 0 else "T4, T6"
        sentences.append(
            f"Significant focal slowing localized to the {affected_side} temporal region ({channels}) "
            f"with regional asymmetry index of {temp_asym:+.2f}."
        )
    elif abs(hemi_asym) >= 0.20:
        dominant_hemi = "left" if hemi_asym > 0 else "right"
        sentences.append(
            f"Hemispheric voltage asymmetry noted with higher amplitude power over the {dominant_hemi} hemisphere."
        )
    else:
        sentences.append("Background activity is broadly symmetric between hemispheres.")

    # 4. Epileptiform Discharges / Sharp Transients
    if kurt >= 12.0:
        sentences.append(
            "Frequent high-amplitude sharp transients and epileptiform-appearing discharges "
            "superimposed on the background."
        )
    elif kurt >= 7.0:
        sentences.append(
            "Occasional sharp wave transients observed, concerning for cortical irritability."
        )
    else:
        sentences.append("No definite epileptiform spike-and-wave discharges identified.")

    # 5. Beta activity
    beta_pct = g_rel["beta"] * 100.0
    if beta_pct >= 20.0:
        sentences.append(
            f"Abundant low-voltage fast beta rhythms ({beta_pct:.1f}%) present over frontocentral regions."
        )

    # 6. Overall Impression
    is_abnormal = (
        (pathological == 1)
        if pathological is not None
        else (slow_idx >= 1.5 or abs(temp_asym) >= 0.25 or kurt >= 8.0 or pdr_alpha < 0.10)
    )
    if is_abnormal:
        reasons = []
        if slow_idx >= 1.5:
            reasons.append("generalized slowing")
        if abs(temp_asym) >= 0.25:
            reasons.append(f"focal {'left' if temp_asym > 0 else 'right'} temporal dysfunction")
        if kurt >= 7.0:
            reasons.append("sharp epileptiform activity")
        if not reasons:
            reasons.append("disorganized background activity")
        reason_str = ", ".join(reasons)
        sentences.append(f"IMPRESSION: Abnormal EEG recording characterized by {reason_str}.")
    else:
        sentences.append("IMPRESSION: Normal EEG recording with intact background organization.")

    # 7. Demographics
    if age is not None and gender is not None:
        g_str = "male" if str(gender).upper().startswith("M") else "female"
        sentences.append(f"The patient is a {age}-year-old {g_str}.")
    elif age is not None:
        sentences.append(f"The patient is {age} years old.")
    elif gender is not None:
        g_str = "male" if str(gender).upper().startswith("M") else "female"
        sentences.append(f"The patient is {g_str}.")

    return " ".join(sentences)


def signal_to_structured_labels(
    window: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    pathological: Optional[int] = None,
) -> Dict[str, int]:
    """
    Computes binary 0/1 structured clinical labels directly from window features.
    Matches STRUCTURED_LABEL_NAMES in data/structured_labels.py.
    """
    metrics = extract_qeeg_metrics(window, fs=fs)
    g_rel = metrics["global_rel"]
    slow_idx = metrics["slowing_index"]
    pdr_alpha = metrics["pdr_alpha_rel"]
    temp_asym = metrics["temporal_asymmetry"]
    hemi_asym = metrics["hemispheric_asymmetry"]
    kurt = metrics["max_kurtosis"]

    has_slowing = int(slow_idx >= 1.35 or (g_rel["delta"] + g_rel["theta"] >= 0.55))
    has_epileptiform = int(kurt >= 8.5)
    focal_left = int(temp_asym >= 0.22 or hemi_asym >= 0.25)
    focal_right = int(temp_asym <= -0.22 or hemi_asym <= -0.25)
    generalized = int(has_slowing and not focal_left and not focal_right)
    bg_normal = int(pdr_alpha >= 0.16 and slow_idx < 1.35 and not has_epileptiform)

    return {
        "slowing": has_slowing,
        "epileptiform": has_epileptiform,
        "focal_left": focal_left,
        "focal_right": focal_right,
        "generalized": generalized,
        "background_normal": bg_normal,
        "delta": int(g_rel["delta"] >= 0.35),
        "theta": int(g_rel["theta"] >= 0.25),
        "alpha": int(g_rel["alpha"] >= 0.18),
        "beta": int(g_rel["beta"] >= 0.15),
    }


if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic 12-second window at 100 Hz (21 channels, 1200 samples)
    t = np.linspace(0, 12, 1200)
    fake_eeg = np.zeros((21, 1200), dtype=np.float32)
    # Add alpha rhythm (10 Hz) in occipital (O1=8, O2=9)
    fake_eeg[8] += 2.0 * np.sin(2 * np.pi * 10.0 * t)
    fake_eeg[9] += 2.0 * np.sin(2 * np.pi * 10.0 * t)
    # Add slow delta (2 Hz) in left temporal (T3=12, T5=14)
    fake_eeg[12] += 4.0 * np.sin(2 * np.pi * 2.0 * t)
    fake_eeg[14] += 4.0 * np.sin(2 * np.pi * 2.0 * t)
    # Add noise
    fake_eeg += 0.5 * np.random.randn(21, 1200).astype(np.float32)

    text = signal_to_clinical_text(fake_eeg, age=42, gender="M", pathological=1)
    labels = signal_to_structured_labels(fake_eeg)
    print("=" * 70)
    print("SIGNAL-TO-TEXT SYNTHESIS TEST:")
    print("=" * 70)
    print(text)
    print("\nSTRUCTURED LABELS:")
    for k, v in labels.items():
        print(f"  {k:20s}: {v}")
    print("=" * 70)
