"""
models/vision_encoder.py
========================
Pretrained Vision Encoder wrapper for EEG-CLIP Multimodal Learning.

WHY
---
To extend EEG-CLIP from (EEG, Text) into a 3-way multimodal space (EEG, Text, Vision),
we project 2D EEG representations (topomaps / spectrograms) through a pretrained
Vision Transformer (ViT-B/16) or ResNet-18/50 backbone into the shared D=64
contrastive embedding space.

Features:
  1. Safe caching to E: drive (avoiding C: drive disk space issues).
  2. Supports ViT-B/16 and lightweight ResNet-18 (ideal for laptop RTX 4050).
  3. Frozen pretrained visual backbone + trainable 3-layer MLP projection head.
  4. L2-normalized output for direct cosine similarity with EEG and Text embeddings.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure weights cache on E: drive
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "E:/eeg_clip/.cache/torch")


class VisionEncoder(nn.Module):
    """
    Wraps a pretrained computer vision backbone (ViT or ResNet)
    and a 3-layer MLP projection head to map 224x224 RGB EEG topomap/spectrogram
    images into the shared contrastive space.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        projection_dim: int = 64,
        proj_hidden_dim: int = 256,
        drop_p: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.freeze_backbone = freeze_backbone
        self.projection_dim = projection_dim

        import torchvision.models as tv_models

        print(f"[vision_encoder] Loading vision backbone: {self.model_name} (pretrained={pretrained})...")

        if "vit" in self.model_name:
            weights = tv_models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.vit_b_16(weights=weights)
            in_dim = 768
            self.backbone.heads = nn.Identity()
        elif "resnet50" in self.model_name:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet50(weights=weights)
            in_dim = 2048
            self.backbone.fc = nn.Identity()
        elif "resnet18" in self.model_name or "resnet" in self.model_name:
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.resnet18(weights=weights)
            in_dim = 512
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported vision model: {model_name}. Use 'resnet18', 'resnet50', or 'vit_b_16'.")

        self.native_dim = in_dim

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print(f"[vision_encoder] Backbone parameters frozen.")

        # 3-layer MLP projection head matching EEG-CLIP paper Sec 3.3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(proj_hidden_dim, projection_dim),
        )

        print(f"[vision_encoder] Initialized: in_dim={in_dim} -> proj_dim={projection_dim}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, 224, 224) float tensor normalized
        returns: (B, projection_dim) unit L2-normalized embedding
        """
        raw_feats = self.backbone(images)  # (B, native_dim)
        proj = self.proj(raw_feats)         # (B, projection_dim)
        return F.normalize(proj, p=2, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)


if __name__ == "__main__":
    print("Testing VisionEncoder...")
    enc = VisionEncoder(model_name="resnet18", pretrained=False, projection_dim=64)
    x = torch.randn(4, 3, 224, 224)
    z = enc(x)
    print("Output shape:", z.shape)
    print("Unit norm check:", z.norm(dim=-1))
