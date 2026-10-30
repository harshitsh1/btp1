"""
models/eeg_encoder.py
======================
WHY
---
The paper uses "Deep4" (Schirrmeister et al., 2017) as the EEG encoder:
"four convolution-max-pooling blocks, using batch normalization and
dropout, followed by a dense softmax classification layer... The output is
flattened and passed to a fully-connected layer to derive a 128-dimensional
embedding" (Sec 3.3).

Deep4Net's defining trick (vs. a generic 1D-CNN) is that the *first* block
factorizes the convolution into a temporal conv (across time only) followed
by a spatial conv (across all EEG channels at once, kernel size =
n_channels), which mirrors how FBCSP-style spatial filtering works. The
next three blocks are plain temporal conv + max-pool blocks operating on
the now-spatially-filtered feature maps.

We reimplement this directly in PyTorch (rather than depending on the
`braindecode` package) so the architecture is fully transparent and easy
to modify, but the design and default hyperparameters follow the original
paper and braindecode's reference implementation.

HOW
---
    from models.eeg_encoder import Deep4EEGEncoder
    enc = Deep4EEGEncoder(n_channels=21, input_window_samples=1200,
                           embed_dim=128)
    z = enc(x)   # x: (batch, n_channels, time) -> z: (batch, embed_dim)

Run standalone for a shape/param-count sanity check:
    python models/eeg_encoder.py
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """One temporal-conv -> batchnorm -> ELU -> maxpool -> dropout block."""

    def __init__(self, in_ch, out_ch, kernel_size, pool_size, pool_stride,
                 drop_p):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1),
                               bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-5)
        self.act = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(pool_size, 1),
                                  stride=(pool_stride, 1))
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


class Deep4EEGEncoder(nn.Module):
    """
    Deep4-style CNN EEG encoder.

    Input:  (batch, n_channels, time)
    Output: (batch, embed_dim)   -- a single embedding per 12s window,
            matching the paper's 128-dimensional EEG embedding fed into
            the contrastive projection head.

    Architecture (paper-faithful "Deep4" 4-block design):
      Block 0 (special "split" first block):
        - temporal conv:  (1 -> 25 filters, kernel 10 along time)
        - spatial conv:   (25 -> 25 filters, kernel = n_channels, across
                            all electrodes at once -> collapses the
                            spatial/channel dimension, like a learned
                            spatial filter bank)
        - BN, ELU, maxpool(3,1), dropout
      Blocks 1-3: temporal conv -> BN -> ELU -> maxpool(3,1) -> dropout,
        channel counts 25->50->100->200 as in the reference Deep4Net.
      Head: adaptive-avg-pool over time, flatten, Linear -> embed_dim.
    """

    def __init__(self, n_channels: int = 21, input_window_samples: int = 1200,
                 embed_dim: int = 128, drop_p: float = 0.5,
                 filters=(25, 25, 50, 100, 200)):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim

        f_time, f_spat, f2, f3, f4 = filters

        # ---- Block 0: temporal then spatial conv ------------------------
        self.temporal_conv = nn.Conv2d(1, f_time, kernel_size=(10, 1), bias=False)
        self.spatial_conv = nn.Conv2d(f_time, f_spat,
                                       kernel_size=(1, n_channels), bias=False)
        self.bn0 = nn.BatchNorm2d(f_spat, momentum=0.1, eps=1e-5)
        self.act0 = nn.ELU()
        self.pool0 = nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        self.drop0 = nn.Dropout(drop_p)

        # ---- Blocks 1-3 ---------------------------------------------------
        self.block1 = _ConvBlock(f_spat, f2, kernel_size=10, pool_size=3,
                                  pool_stride=3, drop_p=drop_p)
        self.block2 = _ConvBlock(f2, f3, kernel_size=10, pool_size=3,
                                  pool_stride=3, drop_p=drop_p)
        self.block3 = _ConvBlock(f3, f4, kernel_size=10, pool_size=3,
                                  pool_stride=3, drop_p=drop_p)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(f4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_channels, time) raw preprocessed EEG window.
        """
        # -> (batch, 1, time, n_channels) so conv2d treats time and
        #    channel-position as the two spatial dims, matching the
        #    temporal-then-spatial factorization above.
        x = x.permute(0, 2, 1).unsqueeze(1)  # (B, 1, T, C)

        x = self.temporal_conv(x)            # (B, f_time, T', C)
        x = self.spatial_conv(x)             # (B, f_spat, T', 1)  (C collapsed)
        x = self.bn0(x)
        x = self.act0(x)
        x = self.pool0(x)
        x = self.drop0(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.gap(x)                      # (B, f4, 1, 1)
        x = x.flatten(1)                     # (B, f4)
        z = self.fc(x)                       # (B, embed_dim)
        return z


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = Deep4EEGEncoder(n_channels=21, input_window_samples=1200, embed_dim=128)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"[eeg_encoder] Deep4EEGEncoder built. Trainable params: {n_params:,}")

    x = torch.randn(8, 21, 1200)  # (batch=8, channels=21, time=1200 @ 100Hz = 12s)
    with torch.no_grad():
        z = enc(x)
    print(f"[eeg_encoder] Input shape:  {tuple(x.shape)}")
    print(f"[eeg_encoder] Output shape: {tuple(z.shape)}  (expected (8, 128))")
    assert z.shape == (8, 128), "Sanity check failed!"
    print("[eeg_encoder] OK.")
