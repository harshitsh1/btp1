"""
models/channel_aware_encoder.py
================================
WHY
---
The original Deep4 EEG encoder collapses all 21 channels with a single
spatial convolution in block 0 -- it learns ONE spatial filter bank but
has no explicit notion of *which* electrode is which or *where* it sits
on the scalp. This means it cannot easily learn "left temporal" vs
"right temporal" distinctions that are critical for clinical EEG.

This module provides a **channel-aware** alternative that:

  1. **Per-channel tokenization**: each of the 21 channels is processed
     by a shared temporal CNN backbone, producing one feature vector per
     channel. This preserves channel identity through the network.

  2. **Electrode position embeddings**: a learnable embedding for each
     of the 21 standard 10-20 electrode positions is added to the
     per-channel features, giving the model explicit spatial information.

  3. **Cross-channel Transformer**: a small Transformer encoder (2-4
     layers) operates over the 21 channel tokens, learning inter-channel
     relationships (e.g. left-right coherence, anterior-posterior
     gradients) via self-attention.

  4. **Window-level attention pooling**: when the model receives multiple
     windows from the same recording, an attention mechanism pools them
     into a single recording-level embedding. During training, all
     windows from one recording are treated as positives for the same
     report text. This replaces the flat per-window contrastive objective
     with a hierarchical one that better matches the recording-level
     report.

Architecture
------------
    Input: (B, 21, 1200)  -- 21 channels, 12s @ 100Hz
    ├── Per-channel temporal CNN: shared 1D-CNN across channels
    │   -> (B, 21, d_model)  -- one d_model-dim feature per channel
    ├── + electrode position embeddings
    │   -> (B, 21, d_model)
    ├── Cross-channel Transformer (2 layers, 4 heads)
    │   -> (B, 21, d_model)
    ├── Channel pooling (mean over channels)
    │   -> (B, d_model)
    └── Linear -> (B, embed_dim=128)

HOW
---
    from models.channel_aware_encoder import ChannelAwareEEGEncoder
    enc = ChannelAwareEEGEncoder(n_channels=21, input_window_samples=1200,
                                  embed_dim=128)
    z = enc(x)  # x: (B, 21, 1200) -> z: (B, 128)

    # With window-level attention pooling:
    from models.channel_aware_encoder import WindowAttentionPooler
    pooler = WindowAttentionPooler(embed_dim=128)
    # window_z: (n_total_windows, 128), window_to_recording: (n_total_windows,)
    recording_z = pooler(window_z, window_to_recording)

Run standalone for a shape/param-count sanity check:
    python models/channel_aware_encoder.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Standard 10-20 electrode positions (2D projected coordinates) ────
# WHY: these approximate 2D scalp positions (azimuth, elevation projected
# to a flat plane) give the model a geometric prior about electrode
# layout. We use them to initialize position embeddings, though the
# embeddings are learnable and can diverge from geometry if helpful.

ELECTRODE_POSITIONS_10_20 = {
    # name: (x, y) normalized to [-1, 1] on the 2D scalp projection
    "FP1": (-0.31, 0.95), "FP2": (0.31, 0.95),
    "F7":  (-0.81, 0.59), "F3":  (-0.35, 0.59),
    "FZ":  (0.00, 0.59),  "F4":  (0.35, 0.59),  "F8": (0.81, 0.59),
    "A1":  (-1.00, 0.00), "T3":  (-0.81, 0.00),
    "C3":  (-0.35, 0.00), "CZ":  (0.00, 0.00),  "C4": (0.35, 0.00),
    "T4":  (0.81, 0.00),  "A2":  (1.00, 0.00),
    "T5":  (-0.81, -0.59), "P3": (-0.35, -0.59),
    "PZ":  (0.00, -0.59),  "P4": (0.35, -0.59), "T6": (0.81, -0.59),
    "O1":  (-0.31, -0.95), "O2": (0.31, -0.95),
}


class TemporalCNNBackbone(nn.Module):
    """
    Shared 1D temporal CNN applied independently to each channel.
    Produces a fixed-dim feature vector from a single channel's time series.

    Architecture: 3 temporal conv blocks with increasing filters,
    followed by adaptive average pooling and a linear layer.
    """

    def __init__(self, input_samples: int = 1200, d_model: int = 128,
                 drop_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: (1, T) -> (32, T/4)
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_p),
            # Block 2: (32, T/4) -> (64, T/16)
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_p),
            # Block 3: (64, T/64) -> (128, T/64)
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(drop_p),
            # Pool to fixed size
            nn.AdaptiveAvgPool1d(1),   # -> (128, 1)
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch * n_channels, 1, time) single-channel time series
        returns: (batch * n_channels, d_model)
        """
        h = self.net(x)          # (B*C, 128, 1)
        h = h.squeeze(-1)       # (B*C, 128)
        return self.proj(h)      # (B*C, d_model)


class ChannelAwareEEGEncoder(nn.Module):
    """
    Channel-aware EEG encoder with electrode position embeddings and
    cross-channel Transformer attention.

    Input:  (batch, n_channels, time)
    Output: (batch, embed_dim)
    """

    def __init__(self, n_channels: int = 21, input_window_samples: int = 1200,
                 embed_dim: int = 128, d_model: int = 128,
                 n_transformer_layers: int = 2, n_heads: int = 4,
                 drop_p: float = 0.3, channel_names: list = None):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.d_model = d_model

        # Per-channel temporal feature extractor (shared weights)
        self.temporal_backbone = TemporalCNNBackbone(
            input_samples=input_window_samples, d_model=d_model, drop_p=drop_p)

        # Learnable electrode position embeddings
        self.electrode_pos_embed = nn.Parameter(
            torch.zeros(n_channels, d_model))
        self._init_position_embeddings(channel_names)

        # Cross-channel Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=drop_p, activation="gelu",
            batch_first=True,
        )
        self.cross_channel_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers)

        # Channel pooling -> embedding
        self.channel_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embed_dim),
        )

    def _init_position_embeddings(self, channel_names):
        """Initialize electrode position embeddings from 10-20 coordinates."""
        if channel_names is None:
            channel_names = list(ELECTRODE_POSITIONS_10_20.keys())[:self.n_channels]

        # Use sinusoidal encoding of (x, y) position for initialization
        with torch.no_grad():
            for i, name in enumerate(channel_names):
                if name in ELECTRODE_POSITIONS_10_20:
                    x, y = ELECTRODE_POSITIONS_10_20[name]
                    # Sinusoidal position encoding from 2D coordinates
                    for j in range(0, self.d_model, 4):
                        freq = 1.0 / (10000 ** (j / self.d_model))
                        self.electrode_pos_embed.data[i, j] = math.sin(x * freq)
                        if j + 1 < self.d_model:
                            self.electrode_pos_embed.data[i, j + 1] = math.cos(x * freq)
                        if j + 2 < self.d_model:
                            self.electrode_pos_embed.data[i, j + 2] = math.sin(y * freq)
                        if j + 3 < self.d_model:
                            self.electrode_pos_embed.data[i, j + 3] = math.cos(y * freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_channels, time) raw preprocessed EEG window.
        returns: (B, embed_dim)
        """
        B, C, T = x.shape

        # Per-channel temporal feature extraction
        # Reshape to (B*C, 1, T) so each channel is processed independently
        x_flat = x.reshape(B * C, 1, T)
        channel_features = self.temporal_backbone(x_flat)  # (B*C, d_model)
        channel_features = channel_features.reshape(B, C, self.d_model)  # (B, C, d_model)

        # Add electrode position embeddings
        channel_features = channel_features + self.electrode_pos_embed.unsqueeze(0)

        # Cross-channel Transformer attention
        channel_features = self.cross_channel_transformer(channel_features)  # (B, C, d_model)

        # Pool over channels (mean) and project to embed_dim
        pooled = channel_features.mean(dim=1)  # (B, d_model)
        z = self.channel_pool(pooled)            # (B, embed_dim)

        return z


class WindowAttentionPooler(nn.Module):
    """
    Attention-based pooling over multiple EEG windows from the same
    recording, producing a single recording-level embedding.

    WHY: each recording is split into many 12-second windows, but
    only one report exists for the whole recording. Rather than
    treating each window independently (and duplicating the report as
    target for each), this module learns to weight windows by
    relevance -- some windows may be more informative (e.g. containing
    epileptiform activity) while others are mostly background.

    Uses a single-head attention mechanism:
        score_i = w^T tanh(V * z_i + b)
        alpha_i = softmax(score_i)
        z_recording = sum_i alpha_i * z_i

    Parameters
    ----------
    embed_dim : int, dimension of per-window embeddings
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, window_embeddings: torch.Tensor,
                recording_ids: torch.Tensor) -> torch.Tensor:
        """
        window_embeddings : (N, D) embeddings for all windows in the batch
        recording_ids     : (N,) integer ID mapping each window to its recording

        Returns
        -------
        pooled : (R, D) one embedding per unique recording
        unique_ids : (R,) the unique recording IDs in order
        """
        unique_ids = torch.unique(recording_ids)
        pooled = []

        for rid in unique_ids:
            mask = recording_ids == rid
            windows = window_embeddings[mask]  # (n_windows, D)

            if windows.shape[0] == 1:
                pooled.append(windows.squeeze(0))
                continue

            # Attention scores
            scores = self.attention(windows)         # (n_windows, 1)
            weights = F.softmax(scores, dim=0)       # (n_windows, 1)
            weighted = (weights * windows).sum(dim=0)  # (D,)
            pooled.append(weighted)

        return torch.stack(pooled, dim=0), unique_ids  # (R, D), (R,)


if __name__ == "__main__":
    print("[channel_aware_encoder] Running sanity checks...")
    torch.manual_seed(0)

    # Default 21 channel names from config
    channel_names = [
        "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
        "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2", "FZ", "CZ", "PZ",
    ]

    # 1. Channel-aware encoder
    enc = ChannelAwareEEGEncoder(
        n_channels=21, input_window_samples=1200, embed_dim=128,
        d_model=128, n_transformer_layers=2, n_heads=4,
        channel_names=channel_names)

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[channel_aware_encoder] Built. Trainable params: {n_params:,}")

    x = torch.randn(4, 21, 1200)
    with torch.no_grad():
        z = enc(x)
    print(f"[channel_aware_encoder] Input:  {tuple(x.shape)}")
    print(f"[channel_aware_encoder] Output: {tuple(z.shape)} (expected (4, 128))")
    assert z.shape == (4, 128), f"Shape mismatch: {z.shape}"

    # 2. Window attention pooler
    pooler = WindowAttentionPooler(embed_dim=128)
    window_z = torch.randn(10, 128)
    rec_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    with torch.no_grad():
        pooled, unique = pooler(window_z, rec_ids)
    print(f"[channel_aware_encoder] WindowPooler: {tuple(window_z.shape)} "
          f"-> {tuple(pooled.shape)} (3 recordings)")
    assert pooled.shape == (3, 128), f"Shape mismatch: {pooled.shape}"

    # 3. Gradient flow check
    x.requires_grad_(True)
    z = enc(x)
    z.sum().backward()
    assert x.grad is not None, "Gradients should flow through encoder"
    print(f"[channel_aware_encoder] Gradient norm: {x.grad.norm().item():.6f}")

    print("[channel_aware_encoder] All checks passed. OK.")
