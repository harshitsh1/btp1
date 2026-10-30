"""
data/eeg_to_image.py
====================
Convert 21-channel EEG windows into 2D vision representations (224x224 RGB images)
for Vision Transformer (ViT) encoding in EEG-CLIP multimodal contrastive learning.

Modes supported:
  1. 'topomap': Multi-band scalp topographic map (R=Delta+Theta slow power,
     G=Alpha rhythm, B=Beta fast power) interpolated on 10-20 electrode positions.
  2. 'spectrogram': Multi-channel time-frequency STFT spectrogram with RGB channels
     representing Left, Midline, and Right hemispheric channels.
  3. 'hybrid': Composite image pairing the spatial topomap and time-frequency spectrogram.
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

CHANNELS_21 = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
]

# Standard 10-20 electrode 2D coordinates on unit disk [-1, 1]
ELECTRODE_COORDS = np.array([
    [-0.30,  0.75],  # FP1
    [ 0.30,  0.75],  # FP2
    [-0.35,  0.42],  # F3
    [ 0.35,  0.42],  # F4
    [-0.40,  0.00],  # C3
    [ 0.40,  0.00],  # C4
    [-0.35, -0.42],  # P3
    [ 0.35, -0.42],  # P4
    [-0.28, -0.75],  # O1
    [ 0.28, -0.75],  # O2
    [-0.72,  0.42],  # F7
    [ 0.72,  0.42],  # F8
    [-0.82,  0.00],  # T3
    [ 0.82,  0.00],  # T4
    [-0.72, -0.42],  # T5
    [ 0.72, -0.42],  # T6
    [-0.92, -0.08],  # A1
    [ 0.92, -0.08],  # A2
    [ 0.00,  0.42],  # FZ
    [ 0.00,  0.00],  # CZ
    [ 0.00, -0.42],  # PZ
], dtype=np.float32)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class TopomapInterpolator:
    """
    Precomputed Gaussian RBF interpolation matrix from 21 electrodes to (H, W).
    Multiplication W @ values takes < 0.5 ms per window.
    """
    def __init__(self, img_size: int = 224, sigma: float = 0.25):
        self.img_size = img_size
        self.sigma = sigma

        y, x = np.mgrid[-1.05:1.05:img_size*1j, -1.05:1.05:img_size*1j].astype(np.float32)
        grid_pts = np.stack([x.ravel(), y.ravel()], axis=1)  # (N, 2)

        # Distance to electrodes: (N, 21)
        dists = np.linalg.norm(grid_pts[:, None, :] - ELECTRODE_COORDS[None, :, :], axis=2)
        weights = np.exp(-dists**2 / (2.0 * sigma**2))

        # Scalp circle mask (radius <= 1.0)
        mask = (grid_pts[:, 0]**2 + grid_pts[:, 1]**2) <= 1.0
        weights[~mask] = 0.0
        row_sums = weights.sum(axis=1, keepdims=True) + 1e-8
        weights = (weights / row_sums) * mask[:, None]

        self.weights = weights.astype(np.float32)  # (H*W, 21)
        self.mask_2d = mask.reshape(img_size, img_size)

        # Draw head boundary and nose marker on scalp
        self.boundary_mask = np.zeros((img_size, img_size), dtype=np.float32)
        radii = np.sqrt(x**2 + y**2)
        circle_edge = (radii >= 0.96) & (radii <= 1.0)
        self.boundary_mask[circle_edge] = 0.4

        # Nose triangle (y > 0.95 and narrow x)
        nose = (y >= 0.98) & (y <= 1.04) & (np.abs(x) <= (1.04 - y) * 0.8)
        self.boundary_mask[nose] = 0.5

    def interpolate(self, channel_values: np.ndarray) -> np.ndarray:
        """
        channel_values: (21, C) where C is number of feature channels (e.g. 3 for RGB).
        Returns: (C, H, W) float32 in [0, 1].
        """
        H = W = self.img_size
        # (H*W, C) = (H*W, 21) @ (21, C)
        grid_vals = self.weights @ channel_values
        grid_vals = grid_vals.reshape(H, W, -1).transpose(2, 0, 1)  # (C, H, W)
        
        # Add head boundary outline
        for c in range(grid_vals.shape[0]):
            grid_vals[c] = np.clip(grid_vals[c] + self.boundary_mask, 0.0, 1.0)
        return grid_vals.astype(np.float32)


# Cache interpolator instances for standard sizes
_INTERPOLATORS = {}

def get_interpolator(img_size: int = 224) -> TopomapInterpolator:
    if img_size not in _INTERPOLATORS:
        _INTERPOLATORS[img_size] = TopomapInterpolator(img_size=img_size)
    return _INTERPOLATORS[img_size]


def eeg_to_topomap(
    window: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    img_size: int = 224,
    normalize_imagenet: bool = True,
) -> np.ndarray:
    """
    Computes 3-channel (RGB) Topomap:
      R: Slow activity (Delta 0.5-4 Hz + Theta 4-8 Hz) relative power
      G: Alpha rhythm (8-13 Hz) relative power
      B: Fast activity (Beta 14-30 Hz) relative power
    Returns: (3, img_size, img_size) float32.
    """
    if isinstance(window, torch.Tensor):
        sig = window.detach().cpu().numpy()
    else:
        sig = np.asarray(window, dtype=np.float32)

    n_samples = sig.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(sig, axis=-1)) ** 2

    # Relative band powers per channel
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total_power = np.sum(fft_vals[:, total_mask], axis=-1, keepdims=True) + 1e-8

    delta_mask = (freqs >= 0.5) & (freqs < 4.0)
    theta_mask = (freqs >= 4.0) & (freqs < 8.0)
    alpha_mask = (freqs >= 8.0) & (freqs < 13.0)
    beta_mask = (freqs >= 14.0) & (freqs < 30.0)

    slow_power = (np.sum(fft_vals[:, delta_mask], axis=-1) + np.sum(fft_vals[:, theta_mask], axis=-1))
    alpha_power = np.sum(fft_vals[:, alpha_mask], axis=-1)
    beta_power = np.sum(fft_vals[:, beta_mask], axis=-1)

    r_vals = slow_power / total_power.squeeze(-1)    # (21,)
    g_vals = alpha_power / total_power.squeeze(-1)   # (21,)
    b_vals = beta_power / total_power.squeeze(-1)    # (21,)

    # Stack to (21, 3) and scale to [0, 1] range
    ch_vals = np.stack([r_vals, g_vals, b_vals], axis=1)  # (21, 3)
    # Contrast stretch per channel to improve discriminability
    ch_vals = np.clip((ch_vals - 0.05) / 0.70, 0.0, 1.0)

    interp = get_interpolator(img_size)
    img = interp.interpolate(ch_vals)  # (3, H, W)

    if normalize_imagenet:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img.astype(np.float32)


def eeg_to_spectrogram(
    window: Union[np.ndarray, torch.Tensor],
    fs: float = 100.0,
    img_size: int = 224,
    normalize_imagenet: bool = True,
) -> np.ndarray:
    """
    Computes a 3-channel time-frequency STFT spectrogram:
      R: Left hemisphere average STFT
      G: Midline channels average STFT
      B: Right hemisphere average STFT
    Returns: (3, img_size, img_size) float32.
    """
    if isinstance(window, torch.Tensor):
        sig = window.detach().cpu().numpy()
    else:
        sig = np.asarray(window, dtype=np.float32)

    # Channel groups
    left_idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16]   # FP1, F3, C3, P3, O1, F7, T3, T5, A1
    mid_idxs = [18, 19, 20]                         # FZ, CZ, PZ
    right_idxs = [1, 3, 5, 7, 9, 11, 13, 15, 17]  # FP2, F4, C4, P4, O2, F8, T4, T6, A2

    # STFT parameters: window length = 128 (1.28s), hop = 16 (0.16s)
    n_fft = 128
    hop_length = 16
    window_hanning = np.hanning(n_fft).astype(np.float32)

    def _channel_stft(ch_indices):
        sig_grp = np.mean(sig[ch_indices], axis=0)  # (n_samples,)
        n_frames = (len(sig_grp) - n_fft) // hop_length + 1
        spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            frame = sig_grp[i * hop_length : i * hop_length + n_fft] * window_hanning
            spec[:, i] = np.abs(np.fft.rfft(frame)) ** 2
        # Crop up to 35 Hz (bin index <= 45 for 100 Hz fs)
        max_bin = int(35.0 / (fs / n_fft))
        spec = spec[:max_bin, :]
        # Log scale
        spec = np.log1p(spec)
        return spec

    spec_left = _channel_stft(left_idxs)
    spec_mid = _channel_stft(mid_idxs)
    spec_right = _channel_stft(right_idxs)

    # Stack to (3, freq_bins, time_frames)
    spec_rgb = np.stack([spec_left, spec_mid, spec_right], axis=0)  # (3, F, T)

    # Normalize each channel to [0, 1]
    for c in range(3):
        p_min, p_max = spec_rgb[c].min(), spec_rgb[c].max()
        spec_rgb[c] = (spec_rgb[c] - p_min) / (p_max - p_min + 1e-6)

    # Resize to (3, img_size, img_size) via bilinear interpolation
    spec_t = torch.from_numpy(spec_rgb).unsqueeze(0)  # (1, 3, F, T)
    spec_resized = F.interpolate(spec_t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    img = spec_resized.squeeze(0).numpy()

    if normalize_imagenet:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

    return img.astype(np.float32)


def eeg_to_image(
    window: Union[np.ndarray, torch.Tensor],
    mode: str = "topomap",
    fs: float = 100.0,
    img_size: int = 224,
    normalize_imagenet: bool = True,
) -> np.ndarray:
    """
    Main entry point: converts a 21-channel EEG window into a (3, img_size, img_size) image.
    mode options: 'topomap', 'spectrogram', 'hybrid'
    """
    if mode == "topomap":
        return eeg_to_topomap(window, fs=fs, img_size=img_size, normalize_imagenet=normalize_imagenet)
    elif mode == "spectrogram":
        return eeg_to_spectrogram(window, fs=fs, img_size=img_size, normalize_imagenet=normalize_imagenet)
    elif mode == "hybrid":
        # Left half = Topomap, Right half = Spectrogram
        half_w = img_size // 2
        topo = eeg_to_topomap(window, fs=fs, img_size=img_size, normalize_imagenet=False)
        spec = eeg_to_spectrogram(window, fs=fs, img_size=img_size, normalize_imagenet=False)
        
        topo_t = torch.from_numpy(topo).unsqueeze(0)
        spec_t = torch.from_numpy(spec).unsqueeze(0)
        
        topo_half = F.interpolate(topo_t, size=(img_size, half_w), mode="bilinear", align_corners=False)
        spec_half = F.interpolate(spec_t, size=(img_size, img_size - half_w), mode="bilinear", align_corners=False)
        
        hybrid = torch.cat([topo_half, spec_half], dim=-1).squeeze(0).numpy()
        if normalize_imagenet:
            hybrid = (hybrid - IMAGENET_MEAN) / IMAGENET_STD
        return hybrid.astype(np.float32)
    else:
        raise ValueError(f"Unknown image mode: {mode}. Choose from 'topomap', 'spectrogram', 'hybrid'.")


def denormalize_imagenet(img: np.ndarray) -> np.ndarray:
    """Reverses ImageNet normalization: (img * std) + mean to [0, 1]."""
    return np.clip((img * IMAGENET_STD) + IMAGENET_MEAN, 0.0, 1.0)


def save_image_to_file(img: np.ndarray, path: str, was_normalized: bool = True):
    """Saves a (3, H, W) float32 image array to disk as a PNG."""
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if was_normalized:
        img_rgb = denormalize_imagenet(img)
    else:
        img_rgb = np.clip(img, 0.0, 1.0)
    arr = (img_rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def compute_full_recording_spectrogram(
    sig: np.ndarray,
    fs: float = 100.0,
    out_path: Optional[str] = None,
    title: str = "EEG Full-Recording Spectrogram",
):
    """
    Computes and plots high-resolution multi-channel STFT spectrograms across
    the entire EDF recording (e.g. 5-30 minutes) broken down into Left, Midline,
    and Right hemispheric groups with calibrated time (minutes) and frequency (0-35 Hz) axes.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    left_idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16]   # Left hemisphere
    mid_idxs = [18, 19, 20]                         # Midline
    right_idxs = [1, 3, 5, 7, 9, 11, 13, 15, 17]  # Right hemisphere

    n_fft = 256
    hop_length = 32
    window_hanning = np.hanning(n_fft).astype(np.float32)

    def _stft(ch_indices):
        grp = np.mean(sig[ch_indices], axis=0)
        n_frames = (len(grp) - n_fft) // hop_length + 1
        spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
        for i in range(n_frames):
            frame = grp[i * hop_length : i * hop_length + n_fft] * window_hanning
            spec[:, i] = np.abs(np.fft.rfft(frame)) ** 2
        max_bin = int(35.0 / (fs / n_fft))
        return np.log1p(spec[:max_bin, :])

    spec_left = _stft(left_idxs)
    spec_mid = _stft(mid_idxs)
    spec_right = _stft(right_idxs)

    duration_min = sig.shape[-1] / (fs * 60.0)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    cmaps = ["Reds", "Greens", "Blues"]
    labels = ["Left Hemisphere (FP1, F3, C3, P3, O1, F7, T3, T5, A1)",
              "Midline Channels (FZ, CZ, PZ)",
              "Right Hemisphere (FP2, F4, C4, P4, O2, F8, T4, T6, A2)"]

    for ax, spec, cmap, lbl in zip(axes, [spec_left, spec_mid, spec_right], cmaps, labels):
        im = ax.imshow(
            spec,
            aspect="auto",
            origin="lower",
            extent=[0, duration_min, 0, 35],
            cmap=cmap,
        )
        ax.set_ylabel("Freq (Hz)", fontsize=10)
        ax.set_title(lbl, fontsize=10, loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax, orientation="vertical", pad=0.01, fraction=0.02)

    axes[-1].set_xlabel("Time (minutes)", fontsize=11)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        plt.close(fig)
        return fig


if __name__ == "__main__":
    import time
    fake_eeg = np.random.randn(21, 1200).astype(np.float32)
    print("Testing eeg_to_image...")
    for mode in ["topomap", "spectrogram", "hybrid"]:
        t0 = time.time()
        img = eeg_to_image(fake_eeg, mode=mode, img_size=224, normalize_imagenet=True)
        t1 = time.time()
        print(f"  Mode '{mode}': shape {img.shape}, min {img.min():.2f}, max {img.max():.2f}, time: {(t1-t0)*1000:.2f} ms")

