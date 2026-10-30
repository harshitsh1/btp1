"""
models/eeg_clip.py
===================
WHY
---
This is the heart of the paper (Sec. 3.3, Fig. 1): take the EEG encoder's
128-dim embedding and the text encoder's (e.g. 768-dim) embedding, project
BOTH into a shared D-dimensional space (D=64 by default, per the best
setting found in Sec. 4.3.2 / Fig. 8) via 3-layer MLP projection heads with
ReLU activations, L2-normalize, and train with the symmetric InfoNCE /
CLIP contrastive loss so that matching (EEG-window, report) pairs end up
close together in that shared space and mismatched pairs end up far apart:

    L = -1/N * sum_i log( exp(sim(x_i,y_i)/tau) / sum_j exp(sim(x_i,y_j)/tau) )

computed in both directions (EEG->text and text->EEG) and averaged, exactly
as in the original CLIP paper (Radford et al., 2021) which this paper
explicitly follows (Sec. 2.2).

HOW
---
    from models.eeg_clip import EEGCLIP, clip_contrastive_loss
    model = EEGCLIP(cfg)
    eeg_z, text_z, logit_scale = model(eeg_batch, text_batch)
    loss = clip_contrastive_loss(eeg_z, text_z, logit_scale)

Run standalone for an end-to-end forward + loss shape/value sanity check
on random data (no internet / pretrained weights needed if you swap in a
tiny local text encoder -- see the __main__ block for a CPU-safe dummy
text encoder fallback used only for this self-test):
    python models/eeg_clip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.eeg_encoder import Deep4EEGEncoder  # noqa: E402


class ProjectionHead(nn.Module):
    """
    3-layer MLP with ReLU activations projecting a modality-specific
    embedding into the shared contrastive space (paper Sec 3.3: "The EEG
    and text embeddings are then fed into MLP projection heads, consisting
    of 3 fully-connected layers with ReLU activations").
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 drop_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _split_decay_params(module):
    """Separates 2D+ weights (linear, conv) from 1D params (bias, LayerNorm, BatchNorm)
    so weight decay does not crush normalization scales to zero."""
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


class EEGCLIP(nn.Module):
    """
    Full dual-encoder model of Figure 1:
        EEG window  -> Deep4EEGEncoder -> ProjectionHead -> eeg_z  (D-dim)
        Report text -> TextEncoder     -> ProjectionHead -> text_z (D-dim)
    Both z's are L2-normalized before the contrastive loss, as in CLIP.

    `logit_scale` is a learned temperature (paper's tau, Sec 2.2),
    parameterized in log-space and exponentiated, exactly like the
    original CLIP implementation, initialized so 1/exp(logit_scale) starts
    at cfg.temperature.
    """

    def __init__(self, cfg, text_encoder: nn.Module = None):
        super().__init__()
        self.cfg = cfg

        self.eeg_encoder = Deep4EEGEncoder(
            n_channels=cfg.n_channels,
            input_window_samples=cfg.window_len_samples,
            embed_dim=cfg.eeg_embed_dim,
        )

        if text_encoder is None:
            from models.text_encoder import TextEncoder
            text_encoder = TextEncoder(
                cfg.text_encoder_name, max_length=cfg.max_text_tokens,
                freeze_n_layers=cfg.freeze_text_encoder_layers,
            )
        self.text_encoder = text_encoder

        self.eeg_proj = ProjectionHead(
            cfg.eeg_embed_dim, cfg.eeg_embed_dim, cfg.projection_dim)
        self.text_proj = ProjectionHead(
            self.text_encoder.hidden_size, self.text_encoder.hidden_size,
            cfg.projection_dim)

        init_logit_scale = torch.log(torch.tensor(1.0 / cfg.temperature))
        self.logit_scale = nn.Parameter(init_logit_scale)

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        z = self.eeg_encoder(eeg)
        z = self.eeg_proj(z)
        return F.normalize(z, dim=-1)

    def encode_text(self, texts, device=None) -> torch.Tensor:
        z = self.text_encoder(texts, device=device)
        z = self.text_proj(z)
        return F.normalize(z, dim=-1)

    def forward(self, eeg: torch.Tensor, texts):
        eeg_z = self.encode_eeg(eeg)
        text_z = self.encode_text(texts, device=eeg.device)
        logit_scale = self.logit_scale.exp()
        return eeg_z, text_z, logit_scale

    def param_groups(self, eeg_lr: float, text_lr_ratio: float,
                      weight_decay: float):
        text_lr = eeg_lr * text_lr_ratio
        eeg_d, eeg_nd = _split_decay_params(self.eeg_encoder)
        eproj_d, eproj_nd = _split_decay_params(self.eeg_proj)
        txt_d, txt_nd = _split_decay_params(self.text_encoder)
        tproj_d, tproj_nd = _split_decay_params(self.text_proj)

        groups = [
            {"params": eeg_d + eproj_d, "lr": eeg_lr, "weight_decay": weight_decay},
            {"params": eeg_nd + eproj_nd, "lr": eeg_lr, "weight_decay": 0.0},
            {"params": txt_d + tproj_d, "lr": text_lr, "weight_decay": weight_decay},
            {"params": txt_nd + tproj_nd, "lr": text_lr, "weight_decay": 0.0},
            {"params": [self.logit_scale], "lr": eeg_lr, "weight_decay": 0.0},
        ]
        print(f"[eeg_clip] Optimizer param groups: EEG branch lr={eeg_lr}, "
              f"text branch lr={text_lr} (ratio={text_lr_ratio}), "
              f"weight_decay={weight_decay}")
        return groups


def clip_contrastive_loss(eeg_z: torch.Tensor, text_z: torch.Tensor,
                           logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE / CLIP loss (paper Eq. in Sec 2.2, Radford et al. 2021):

        L = 1/2 * [ CE(logits_eeg2text, labels) + CE(logits_text2eeg, labels) ]

    where logits = logit_scale * eeg_z @ text_z.T and labels = arange(N)
    (the diagonal is the only correct match within the batch -- this is
    why the paper trains with batch_size=64: enough negatives per step for
    the contrastive objective to be informative).
    """
    logits_per_eeg = logit_scale * eeg_z @ text_z.t()      # (B, B)
    logits_per_text = logits_per_eeg.t()

    batch_size = eeg_z.shape[0]
    labels = torch.arange(batch_size, device=eeg_z.device)

    loss_e2t = F.cross_entropy(logits_per_eeg, labels)
    loss_t2e = F.cross_entropy(logits_per_text, labels)
    return (loss_e2t + loss_t2e) / 2.0


# =====================================================================
# EEG-CLIP++ : extended model with all four improvements
# =====================================================================

class EEGCLIPPlusPlus(EEGCLIP):
    """
    EEG-CLIP++ extends the base EEGCLIP model with four improvements:

      1. **SigLIP sigmoid / soft-target loss**: replaces InfoNCE to avoid
         false-negative pressure from near-duplicate reports.

      2. **Structured-finding auxiliary heads**: multi-label classification
         heads (slowing, epileptiform, lateralization, etc.) trained jointly
         with the contrastive loss for richer EEG encoder supervision.

      3. **Channel-aware EEG encoder**: per-channel tokenization with
         electrode position embeddings and cross-channel Transformer,
         replacing the monolithic Deep4 spatial convolution.

      4. **CoOp-style learned prompts**: learnable continuous prompt
         embeddings for zero-shot classification (used at eval time).

    All improvements are gated by cfg flags (`cfg.loss_type`,
    `cfg.use_auxiliary_heads`, `cfg.use_channel_aware_encoder`,
    `cfg.use_learned_prompts`) so you can ablate any subset.

    Backward compatible: when all flags are off / set to "infonce",
    this reduces exactly to the original EEGCLIP.
    """

    def __init__(self, cfg, text_encoder: nn.Module = None):
        # Call parent __init__ which builds the original model components
        super().__init__(cfg, text_encoder=text_encoder)

        # ── Improvement 3: channel-aware encoder ──
        if cfg.use_channel_aware_encoder:
            from models.channel_aware_encoder import ChannelAwareEEGEncoder
            print("[eeg_clip++] Using channel-aware EEG encoder with "
                  f"{cfg.channel_encoder_n_layers} Transformer layers, "
                  f"{cfg.channel_encoder_n_heads} heads")
            self.eeg_encoder = ChannelAwareEEGEncoder(
                n_channels=cfg.n_channels,
                input_window_samples=cfg.window_len_samples,
                embed_dim=cfg.eeg_embed_dim,
                d_model=cfg.channel_encoder_d_model,
                n_transformer_layers=cfg.channel_encoder_n_layers,
                n_heads=cfg.channel_encoder_n_heads,
                channel_names=cfg.channels_21,
            )
            # Rebuild EEG projection head for the new encoder's output dim
            self.eeg_proj = ProjectionHead(
                cfg.eeg_embed_dim, cfg.eeg_embed_dim, cfg.projection_dim)

        # ── Improvement 1: SigLIP loss needs a learnable bias ──
        if cfg.loss_type == "sigmoid":
            self.sigmoid_bias = nn.Parameter(torch.tensor(0.0))
            print("[eeg_clip++] Using SigLIP sigmoid contrastive loss")
        elif cfg.loss_type == "soft_target":
            print("[eeg_clip++] Using soft-target contrastive loss "
                  f"(temperature={cfg.soft_target_temperature})")
        else:
            print("[eeg_clip++] Using original InfoNCE contrastive loss")

        # ── Improvement 2: auxiliary classification heads ──
        if cfg.use_auxiliary_heads:
            from models.losses import AuxiliaryHeadLoss
            self.auxiliary_heads = AuxiliaryHeadLoss(
                input_dim=cfg.eeg_embed_dim,
                label_names=cfg.structured_label_names,
                hidden_dim=cfg.aux_hidden_dim,
            )
            print(f"[eeg_clip++] Auxiliary heads on EEG embedding: "
                  f"{cfg.structured_label_names}")
        else:
            self.auxiliary_heads = None

        # ── Improvement 3b: window attention pooler ──
        if cfg.use_window_pooling:
            from models.channel_aware_encoder import WindowAttentionPooler
            self.window_pooler = WindowAttentionPooler(
                embed_dim=cfg.eeg_embed_dim)
            print("[eeg_clip++] Window-level attention pooling enabled")
        else:
            self.window_pooler = None

    def compute_contrastive_loss(self, eeg_z, text_z, logit_scale):
        """Select the appropriate contrastive loss based on config."""
        if self.cfg.loss_type == "sigmoid":
            from models.losses import sigmoid_contrastive_loss
            bias = self.sigmoid_bias if hasattr(self, 'sigmoid_bias') else None
            return sigmoid_contrastive_loss(eeg_z, text_z, logit_scale, bias)
        elif self.cfg.loss_type == "soft_target":
            from models.losses import soft_target_contrastive_loss
            return soft_target_contrastive_loss(
                eeg_z, text_z, logit_scale,
                temperature_soft=self.cfg.soft_target_temperature)
        else:
            return clip_contrastive_loss(eeg_z, text_z, logit_scale)

    def compute_auxiliary_loss(self, eeg_raw_embedding, structured_labels):
        """Compute auxiliary head loss on raw EEG embeddings."""
        if self.auxiliary_heads is None:
            return torch.tensor(0.0, device=eeg_raw_embedding.device)
        return self.auxiliary_heads(eeg_raw_embedding, structured_labels)

    def forward_with_loss(self, eeg, texts, structured_labels=None, images=None):
        """
        Full forward pass + combined loss computation.

        Parameters
        ----------
        eeg : (B, C, T) EEG tensor
        texts : list[str] of length B
        structured_labels : dict[str, Tensor(B)] optional structured labels

        Returns
        -------
        loss_total : scalar combined loss
        loss_dict  : dict with individual loss components for logging
        """
        # Encode EEG (get both raw embedding and projected/normalized)
        eeg_raw = self.eeg_encoder(eeg)          # (B, eeg_embed_dim)
        eeg_z = self.eeg_proj(eeg_raw)           # (B, projection_dim)
        eeg_z = F.normalize(eeg_z, dim=-1)

        # Encode text
        text_z = self.encode_text(texts, device=eeg.device)
        logit_scale = self.logit_scale.exp()

        # ── Contrastive loss ──
        loss_contrastive = self.compute_contrastive_loss(
            eeg_z, text_z, logit_scale)

        # ── Auxiliary loss ──
        loss_aux = torch.tensor(0.0, device=eeg.device)
        if self.cfg.use_auxiliary_heads and structured_labels is not None:
            loss_aux = self.compute_auxiliary_loss(eeg_raw, structured_labels)

        # ── Combined ──
        loss_total = loss_contrastive + self.cfg.aux_loss_weight * loss_aux

        loss_dict = {
            "contrastive": loss_contrastive.item(),
            "auxiliary": loss_aux.item(),
            "total": loss_total.item(),
            "logit_scale": logit_scale.item(),
        }

        return loss_total, loss_dict

    def param_groups(self, eeg_lr: float, text_lr_ratio: float,
                      weight_decay: float):
        """
        Extended param groups for EEG-CLIP++.
        Protects 1D normalization and bias parameters from weight decay.
        """
        text_lr = eeg_lr * text_lr_ratio
        eeg_d, eeg_nd = _split_decay_params(self.eeg_encoder)
        eproj_d, eproj_nd = _split_decay_params(self.eeg_proj)
        txt_d, txt_nd = _split_decay_params(self.text_encoder)
        tproj_d, tproj_nd = _split_decay_params(self.text_proj)

        groups = [
            {"params": eeg_d + eproj_d, "lr": eeg_lr, "weight_decay": weight_decay},
            {"params": eeg_nd + eproj_nd, "lr": eeg_lr, "weight_decay": 0.0},
            {"params": txt_d + tproj_d, "lr": text_lr, "weight_decay": weight_decay},
            {"params": txt_nd + tproj_nd, "lr": text_lr, "weight_decay": 0.0},
            {"params": [self.logit_scale], "lr": eeg_lr, "weight_decay": 0.0},
        ]

        # Auxiliary heads (at EEG LR, with no-decay for bias/norm)
        if self.auxiliary_heads is not None:
            aux_d, aux_nd = _split_decay_params(self.auxiliary_heads)
            if aux_d:
                groups.append({"params": aux_d, "lr": eeg_lr, "weight_decay": weight_decay})
            if aux_nd:
                groups.append({"params": aux_nd, "lr": eeg_lr, "weight_decay": 0.0})

        # SigLIP bias (scalar, no weight decay)
        if hasattr(self, 'sigmoid_bias'):
            groups.append({
                "params": [self.sigmoid_bias],
                "lr": eeg_lr, "weight_decay": 0.0,
            })

        # Window pooler
        if self.window_pooler is not None:
            wp_d, wp_nd = _split_decay_params(self.window_pooler)
            if wp_d:
                groups.append({"params": wp_d, "lr": eeg_lr, "weight_decay": weight_decay})
            if wp_nd:
                groups.append({"params": wp_nd, "lr": eeg_lr, "weight_decay": 0.0})

        print(f"[eeg_clip++] Optimizer param groups: EEG branch lr={eeg_lr}, "
              f"text branch lr={text_lr} (ratio={text_lr_ratio}), "
              f"weight_decay={weight_decay}, "
              f"n_groups={len(groups)}")
        return groups


class EEGCLIPMultimodal(EEGCLIPPlusPlus):
    """
    Multimodal 3-Way EEG-CLIP Model:
      - EEG Branch: Deep4 / ChannelAware -> EEG Proj Head (64-dim)
      - Text Branch: BERT TextEncoder -> Text Proj Head (64-dim)
      - Vision Branch: ViT / ResNet -> Vision Proj Head (64-dim)
    Optimizes 3-way contrastive alignment (EEG<->Text, EEG<->Vision, Text<->Vision)
    plus auxiliary heads for clinical structured findings.
    """

    def __init__(self, cfg, text_encoder=None, vision_encoder=None):
        super().__init__(cfg, text_encoder=text_encoder)
        self.vision_encoder = vision_encoder
        if self.vision_encoder is None and getattr(cfg, "use_vision_encoder", False):
            from models.vision_encoder import VisionEncoder
            self.vision_encoder = VisionEncoder(
                model_name=getattr(cfg, "vision_encoder_name", "resnet18"),
                pretrained=True,
                freeze_backbone=getattr(cfg, "vision_freeze_backbone", True),
                projection_dim=cfg.projection_dim,
            )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder is not initialized. Set cfg.use_vision_encoder=True")
        return self.vision_encoder(images)

    def forward_with_loss(self, eeg: torch.Tensor, texts: list,
                          structured_labels: dict = None,
                          images: torch.Tensor = None):
        if images is None or self.vision_encoder is None:
            return super().forward_with_loss(eeg, texts, structured_labels=structured_labels)

        # 1. Encode EEG
        eeg_raw = self.eeg_encoder(eeg)
        eeg_z = self.eeg_proj(eeg_raw)
        eeg_z = F.normalize(eeg_z, dim=-1)

        # 2. Encode text
        text_z = self.encode_text(texts, device=eeg.device)

        # 3. Encode vision
        vision_z = self.encode_vision(images.to(eeg.device))

        logit_scale = self.logit_scale.exp()

        # 4. Multi-way contrastive loss
        loss_eeg_text = self.compute_contrastive_loss(eeg_z, text_z, logit_scale)
        loss_eeg_vision = self.compute_contrastive_loss(eeg_z, vision_z, logit_scale)
        loss_text_vision = self.compute_contrastive_loss(text_z, vision_z, logit_scale)

        w_v = getattr(self.cfg, "vision_loss_weight", 1.0)
        w_tv = getattr(self.cfg, "text_vision_loss_weight", 0.5)

        loss_contrastive = loss_eeg_text + w_v * loss_eeg_vision + w_tv * loss_text_vision

        # 5. Auxiliary loss
        loss_aux = torch.tensor(0.0, device=eeg.device)
        if self.cfg.use_auxiliary_heads and structured_labels is not None:
            loss_aux = self.compute_auxiliary_loss(eeg_raw, structured_labels)

        # 6. Combined total loss
        loss_total = loss_contrastive + self.cfg.aux_loss_weight * loss_aux

        loss_dict = {
            "total": loss_total.item(),
            "contrastive": loss_contrastive.item(),
            "eeg_text": loss_eeg_text.item(),
            "eeg_vision": loss_eeg_vision.item(),
            "text_vision": loss_text_vision.item(),
            "auxiliary": loss_aux.item(),
            "logit_scale": logit_scale.item(),
        }
        return loss_total, loss_dict

    def param_groups(self, eeg_lr: float, text_lr_ratio: float, weight_decay: float):
        groups = super().param_groups(eeg_lr, text_lr_ratio, weight_decay)
        if self.vision_encoder is not None:
            v_lr = eeg_lr * getattr(self.cfg, "vision_lr_ratio", 1e-3)
            v_d, v_nd = _split_decay_params(self.vision_encoder)
            if v_d:
                groups.append({"params": v_d, "lr": v_lr, "weight_decay": weight_decay})
            if v_nd:
                groups.append({"params": v_nd, "lr": v_lr, "weight_decay": 0.0})
            print(f"[eeg_clip_multimodal] Vision encoder param groups: lr={v_lr}, "
                  f"{len(v_d)} decay tensors, {len(v_nd)} no-decay tensors")
        return groups


if __name__ == "__main__":
    print("=" * 70)
    print("[eeg_clip] Running EEG-CLIP & EEG-CLIP++ self-tests")
    print("=" * 70)

    # ── Original EEGCLIP test (unchanged) ──
    print("\n[1/2] Original EEGCLIP (CPU, dummy text encoder)...")

    class _DummyTextEncoder(nn.Module):
        """CPU-only stand-in for TextEncoder, used ONLY for this self-test
        so the file can be sanity-checked without network access."""
        hidden_size = 32

        def __init__(self):
            super().__init__()
            self.embed = nn.EmbeddingBag(1000, self.hidden_size)

        def forward(self, texts, device=None):
            device = device or torch.device("cpu")
            idxs = [torch.randint(0, 1000, (5,)) for _ in texts]
            flat = torch.cat(idxs).to(device)
            offsets = torch.tensor(
                [0] + [len(i) for i in idxs[:-1]], device=device
            ).cumsum(0)
            return self.embed(flat, offsets)

    from config import cfg
    cfg.n_channels = 21
    cfg.window_len_samples = 1200
    cfg.projection_dim = 64

    # Disable ++ features for baseline test
    cfg.use_channel_aware_encoder = False
    cfg.use_auxiliary_heads = False
    cfg.loss_type = "infonce"

    model = EEGCLIP(cfg, text_encoder=_DummyTextEncoder())
    eeg = torch.randn(6, 21, 1200)
    texts = ["report " + str(i) for i in range(6)]

    eeg_z, text_z, scale = model(eeg, texts)
    print(f"[eeg_clip] eeg_z: {tuple(eeg_z.shape)}, "
          f"text_z: {tuple(text_z.shape)}, logit_scale: {scale.item():.3f}")

    loss = clip_contrastive_loss(eeg_z, text_z, scale)
    print(f"[eeg_clip] Contrastive loss (random init): {loss.item():.4f}")
    print("[eeg_clip] Original EEGCLIP OK.\n")

    # ── EEGCLIPPlusPlus test ──
    print("[2/2] EEGCLIPPlusPlus (CPU, dummy text encoder)...")

    cfg.loss_type = "sigmoid"
    cfg.use_auxiliary_heads = True
    cfg.use_channel_aware_encoder = True
    cfg.use_learned_prompts = False  # needs real text encoder
    cfg.use_window_pooling = False
    cfg.eeg_embed_dim = 32
    cfg.channel_encoder_d_model = 32

    model_pp = EEGCLIPPlusPlus(cfg, text_encoder=_DummyTextEncoder())
    eeg = torch.randn(4, 21, 1200)
    texts = ["report " + str(i) for i in range(4)]
    struct_labels = {
        "slowing": torch.tensor([1, 0, -1, 1]),
        "epileptiform": torch.tensor([0, 1, 0, -1]),
    }

    loss_total, loss_dict = model_pp.forward_with_loss(eeg, texts, struct_labels)
    print(f"[eeg_clip++] Loss breakdown: {loss_dict}")
    loss_total.backward()
    print("[eeg_clip++] Backward pass OK.")

    n_params = sum(p.numel() for p in model_pp.parameters())
    n_trainable = sum(p.numel() for p in model_pp.parameters()
                      if p.requires_grad)
    print(f"[eeg_clip++] Params: {n_params:,} total, "
          f"{n_trainable:,} trainable")
    print("[eeg_clip++] EEGCLIPPlusPlus OK.")

