"""
EEG-CLIP: Contrastive EEG-Text Learning 
Requires: torch, transformers, mne, numpy
"""

import os
os.environ["HF_HOME"] = r"E:\huggingface_cache"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
import mne

CACHE_DIR = r"E:\huggingface_cache"



# ============================================================================
# 1.  EEG ENCODER (Conformer-style:  CNN + Time Attention)
# ============================================================================

class SqueezeExcite1D(nn.Module):
    """ channel attention for 1D signals."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class DepthwiseSeparableConv1D(nn.Module):
    """ efficient convolution: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size, stride, padding, 
            groups=in_ch, bias=bias
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_ch)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """Residual block with depthwise separable conv + SE + GELU."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv = DepthwiseSeparableConv1D(
            in_ch, out_ch, kernel_size, stride, padding
        )
        self.se = SqueezeExcite1D(out_ch)
        self.act = nn.GELU()
        
        # Residual connection
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        x = self.conv(x)
        x = self.se(x)
        x = self.act(x)
        return x + residual


class EEGEncoder(nn.Module):
    """
     EEG encoder using:
    - Progressive strided convolutions (efficient downsampling)
    - Squeeze-and-Excitation channel attention
    - Multi-head self-attention over time (Conformer-style)
    - LayerNorm in the transformer portion
    """
    def __init__(self, n_channels=21, embed_dim=128):
        super().__init__()
        
        # Stem: large kernel, stride 2 for initial downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        #  convolutional backbone (total stride = 2 * 2 * 2 * 2 = 16)
        # At 100Hz x 12s = 1200 samples -> after stem: 600 -> after stages: ~37 time steps
        self.stage1 = ConvBlock(64, 64, kernel_size=15, stride=2)    # /4
        self.stage2 = ConvBlock(64, 128, kernel_size=9, stride=2)   # /8
        self.stage3 = ConvBlock(128, 256, kernel_size=5, stride=2)  # /16
        self.stage4 = ConvBlock(256, 512, kernel_size=3, stride=1)  # /16 (keep stride)
        
        # : Conformer-style time attention (attention over temporal dimension)
        self.time_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(512)
        self.attn_dropout = nn.Dropout(0.1)
        
        # Global pooling + projection to shared embedding space
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        #  projection head with LayerNorm and GELU
        self.project = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
    
    def forward(self, x):
        """
        x: (batch, n_channels, time_samples)
        """
        # Convolutional feature extraction
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)  # (B, 512, T')
        
        # Conformer-style time attention: model long-range temporal dependencies
        # Permute to (B, T, C) for transformer
        x_t = x.permute(0, 2, 1)
        x_t = self.attn_norm(x_t)
        attn_out, _ = self.time_attn(x_t, x_t, x_t)
        x_t = x_t + self.attn_dropout(attn_out)  # residual
        x = x_t.permute(0, 2, 1)  # back to (B, C, T)
        
        # Global average pooling
        x = self.pool(x).squeeze(-1)  # (B, 512)
        x = self.project(x)  # (B, embed_dim)
        return x


# ============================================================================
# 2.  TEXT ENCODER (DeBERTa-v3 instead of BERT)
# ============================================================================

class TextEncoder(nn.Module):
    """
    Uses DeBERTa-v3-base: disentangled attention, improved pretraining over BERT.
    Alternative: 'answerdotai/BERT-base' (2024), 'BAAI/bge-large-en-v1.5'
    """
    def __init__(self, model_name="microsoft/deberta-v3-base", embed_dim=128):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        
        #  projection head
        self.project = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, embed_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False
        )
        
        #  pooling: mean of last hidden states weighted by attention mask
        # (more robust than just CLS token)
        last_hidden = outputs.last_hidden_state  # (B, L, H)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
        mean_embeddings = sum_embeddings / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        x = self.project(mean_embeddings)
        return x


# ============================================================================
# 3. EEG-CLIP MODEL (Contrastive Learning)
# ============================================================================

class EEGCLIP(nn.Module):
    def __init__(
        self, 
        n_channels=21, 
        embed_dim=128, 
        text_model="microsoft/deberta-v3-base",
        init_temperature=0.07
    ):
        super().__init__()
        self.eeg_encoder = EEGEncoder(n_channels, embed_dim)
        self.text_encoder = TextEncoder(text_model, embed_dim)
        
        # Learnable temperature parameter ( CLIP practice)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / init_temperature))
    
    def forward(self, eeg, text_input_ids, text_attention_mask):
        eeg_embed = self.eeg_encoder(eeg)
        text_embed = self.text_encoder(text_input_ids, text_attention_mask)
        
        # L2 normalize embeddings (cosine similarity space)
        eeg_embed = F.normalize(eeg_embed, p=2, dim=-1)
        text_embed = F.normalize(text_embed, p=2, dim=-1)
        
        # Learnable temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # Similarity matrix
        logits_per_eeg = logit_scale * eeg_embed @ text_embed.T
        logits_per_text = logits_per_eeg.T
        
        return logits_per_eeg, logits_per_text, eeg_embed, text_embed


def clip_loss(logits_per_eeg, logits_per_text):
    """Symmetric cross-entropy contrastive loss."""
    batch_size = logits_per_eeg.shape[0]
    labels = torch.arange(batch_size, device=logits_per_eeg.device)
    
    loss_eeg = F.cross_entropy(logits_per_eeg, labels)
    loss_text = F.cross_entropy(logits_per_text, labels)
    return (loss_eeg + loss_text) / 2


# ============================================================================
# 4. PREPROCESSING (TUH EEG -> Model Input)
# ============================================================================

# 21 channels from the 10-20 system used in the paper
CHANNELS_21 = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
    'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
    'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
    'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
    'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
]


def preprocess_tuh_eeg(raw, target_sfreq=100):
    """
    Preprocess raw TUH EEG file following EEG-CLIP paper specs with  additions.
    """
    # 1. Select 21 standard channels (ignore EKG, photic, etc.)
    available_chs = [ch for ch in CHANNELS_21 if ch in raw.ch_names]
    if len(available_chs) < 21:
        missing = set(CHANNELS_21) - set(raw.ch_names)
        print(f"Warning: Missing channels: {missing}")
    
    raw = raw.pick_channels(available_chs)
    raw = raw.reorder_channels([ch for ch in CHANNELS_21 if ch in raw.ch_names])
    
    # 2.  addition: high-pass filter to remove slow drifts
    raw = raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin')
    
    # 3. Resample to 100 Hz (paper specification)
    raw = raw.resample(target_sfreq)
    
    # 4. Get data in microvolts
    data = raw.get_data(units='uV')  # (n_channels, n_times)
    
    # 5. Clip amplitude to ±800 μV (paper)
    data = np.clip(data, -800, 800)
    
    # 6. Scale to approximate unit variance (paper)
    data = data / 30.0
    
    return data


def create_windows(data, sfreq=100, window_sec=12, stride_sec=5.19):
    """
    Create overlapping windows. Paper uses 1200 samples (12s) with stride 519.
    """
    window_samples = int(window_sec * sfreq)
    stride_samples = int(stride_sec * sfreq)
    
    windows = []
    for start in range(0, data.shape[1] - window_samples + 1, stride_samples):
        win = data[:, start:start + window_samples]
        if win.shape[1] == window_samples:
            windows.append(win)
    
    return np.array(windows, dtype=np.float32)  # (n_windows, n_channels, n_samples)


# ============================================================================
# 5. USAGE EXAMPLE
# ============================================================================

def main():
    # Load your TUH file
    raw = mne.io.read_raw_edf(
        r"E:\tuh_eeg\v3.0.1\edf\eval\abnormal\01_tcp_ar\aaaaabdo_s003_t000.edf",
        preload=True,
        verbose=False
    )
    
    # Preprocess
    data = preprocess_tuh_eeg(raw, target_sfreq=100)
    windows = create_windows(data, sfreq=100)  # (n_windows, 21, 1200)
    
    print(f"Preprocessed windows shape: {windows.shape}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EEGCLIP(n_channels=21, embed_dim=128).to(device)
    
    # Dummy forward pass
    batch_size = 4
    eeg_batch = torch.from_numpy(windows[:batch_size]).to(device)  # (4, 21, 1200)
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    texts = [
        "This is a normal EEG recording with posterior dominant rhythm.",
        "This EEG shows abnormal epileptiform discharges.",
        "This is a normal recording",
        "This is an abnormal recording"
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, 
                       max_length=128, return_tensors="pt").to(device)
    
    logits_eeg, logits_text, eeg_emb, text_emb = model(
        eeg_batch, encoded['input_ids'], encoded['attention_mask']
    )
    
    print(f"EEG embeddings shape: {eeg_emb.shape}")      # (4, 128)
    print(f"Text embeddings shape: {text_emb.shape}")    # (4, 128)
    print(f"Logit scale: {model.logit_scale.exp().item():.3f}")
    
    # Compute loss
    loss = clip_loss(logits_eeg, logits_text)
    print(f"Contrastive loss: {loss.item():.4f}")
    
    # Zero-shot similarity: EEG vs text prompts
    similarity = (eeg_emb @ text_emb.T).cpu().detach().numpy()
    print("\nEEG-Text Similarity Matrix:")
    print(similarity)


if __name__ == "__main__":
    main()