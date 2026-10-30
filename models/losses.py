"""
models/losses.py
=================
WHY
---
The original EEG-CLIP uses symmetric InfoNCE (CLIP) loss, which treats
every in-batch off-diagonal pair as a hard negative. In EEG-CLIP, many
reports say nearly identical things (e.g. "normal EEG"), so these
"negatives" are really false negatives -- the loss actively pushes apart
recordings that should be close. Two fixes implemented here:

  1. **SigLIP sigmoid loss** (Zhai et al., 2023): replaces the softmax
     cross-entropy with per-pair binary sigmoid loss. Each (i,j) pair is
     classified independently as matching (1) or not (0), so a
     near-duplicate report paired with a different EEG doesn't compete
     in a softmax denominator. This removes false-negative pressure
     without requiring any text-similarity computation.

  2. **Soft-target contrastive loss**: keeps InfoNCE structure but
     replaces the hard one-hot target (diagonal = 1, off-diag = 0) with
     a soft target matrix built from pairwise text-embedding cosine
     similarity. Near-duplicate reports get a label close to 1 instead
     of 0, so the model is not penalized for mapping their EEGs nearby.

Both losses are drop-in replacements for `clip_contrastive_loss` in
models/eeg_clip.py -- same signature `(eeg_z, text_z, logit_scale)`,
same scalar output.

HOW
---
    from models.losses import sigmoid_contrastive_loss, soft_target_contrastive_loss
    loss = sigmoid_contrastive_loss(eeg_z, text_z, logit_scale, bias)
    # or
    loss = soft_target_contrastive_loss(eeg_z, text_z, logit_scale)

Run standalone for a sanity check:
    python models/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_contrastive_loss(eeg_z: torch.Tensor, text_z: torch.Tensor,
                              logit_scale: torch.Tensor,
                              bias: torch.Tensor = None) -> torch.Tensor:
    """
    SigLIP-style sigmoid contrastive loss (Zhai et al., 2023).

    Instead of softmax-based InfoNCE, each (i,j) pair is treated as an
    independent binary classification: is this pair a true match?

        L = -1/n * sum_i sum_j [
              y_ij * log(sigma(z_i . z_j * t - b))
            + (1 - y_ij) * log(sigma(-(z_i . z_j * t - b)))
        ]

    where y_ij = 1 if i == j (matching pair), 0 otherwise,
    t = logit_scale (learned), b = bias (learned, optional).

    WHY this helps: in InfoNCE, false negatives (near-duplicate reports
    paired with different EEGs) compete in the softmax denominator and
    actively push the model to separate things that should be close.
    The sigmoid loss has no denominator -- each pair is scored
    independently, so false negatives only add a small constant penalty
    instead of distorting the entire gradient landscape.

    Parameters
    ----------
    eeg_z  : (B, D) L2-normalized EEG embeddings
    text_z : (B, D) L2-normalized text embeddings
    logit_scale : scalar (exp of learned log-temperature)
    bias   : scalar (learned bias, optional -- if None, defaults to 0)
    """
    # Pairwise similarity matrix: (B, B)
    logits = logit_scale * eeg_z @ text_z.t()
    if bias is not None:
        logits = logits - bias

    batch_size = eeg_z.shape[0]

    # Target matrix: 1 on diagonal (true matches), -1 off-diagonal.
    # To handle in-batch identical/near-duplicate texts (e.g. synthetic templates),
    # pairs sharing identical text embeddings are treated as positive (+1)
    # rather than penalizing the model for mapping them to the same concept.
    with torch.no_grad():
        text_sim = text_z @ text_z.t()
        pos_mask = (text_sim > 0.99) | torch.eye(batch_size, dtype=torch.bool, device=eeg_z.device)
        labels = 2 * pos_mask.float() - 1

    # Standard SigLIP normalization (normalized by batch size, not B^2)
    loss = -F.logsigmoid(labels * logits).sum() / batch_size
    return loss


def soft_target_contrastive_loss(eeg_z: torch.Tensor, text_z: torch.Tensor,
                                  logit_scale: torch.Tensor,
                                  temperature_soft: float = 0.05
                                  ) -> torch.Tensor:
    """
    InfoNCE with soft targets derived from text-embedding similarity.

    WHY: standard CLIP uses hard one-hot targets (only the diagonal pair
    is "correct"). But when two reports are nearly identical, pushing
    their EEGs apart is counterproductive. This variant builds a soft
    target distribution from text-to-text cosine similarity:

        soft_target_ij = softmax(text_sim_ij / tau_soft)

    so near-duplicate reports produce a near-uniform distribution instead
    of a sharp one-hot, and the KL divergence against the model's logits
    is minimal even if those EEGs are mapped close together.

    Parameters
    ----------
    eeg_z  : (B, D) L2-normalized EEG embeddings
    text_z : (B, D) L2-normalized text embeddings
    logit_scale : scalar
    temperature_soft : float, temperature for softening text-similarity targets
    """
    # Model's predicted similarity
    logits_e2t = logit_scale * eeg_z @ text_z.t()  # (B, B)
    logits_t2e = logits_e2t.t()

    # Soft targets from text-text similarity
    with torch.no_grad():
        text_sim = text_z @ text_z.t()  # (B, B) cosine similarity
        # Sharpen with a temperature and convert to probability via softmax
        soft_targets = F.softmax(text_sim / temperature_soft, dim=-1)

    # KL divergence: sum_j soft_target_j * log(soft_target_j / pred_j)
    # = CE(soft_target, pred) - H(soft_target)
    # Since H(soft_target) is constant w.r.t. model params, we just use CE.
    loss_e2t = (-soft_targets * F.log_softmax(logits_e2t, dim=-1)).sum(dim=-1).mean()
    loss_t2e = (-soft_targets * F.log_softmax(logits_t2e, dim=-1)).sum(dim=-1).mean()

    return (loss_e2t + loss_t2e) / 2.0


class AuxiliaryHeadLoss(nn.Module):
    """
    Multi-label auxiliary classification heads on EEG embeddings.

    WHY: structured clinical labels (slowing, epileptiform discharges,
    lateralization, etc.) extracted from reports provide precise,
    clinically meaningful supervision beyond the noisy free-text
    contrastive signal. Training auxiliary heads alongside the contrastive
    loss forces the EEG embedding to encode clinically relevant features
    explicitly, improving both zero-shot and few-shot downstream
    performance.

    Each head is a small 2-layer MLP with binary cross-entropy loss.
    Labels with value -1 (unknown/unparseable) are masked out.
    """

    def __init__(self, input_dim: int, label_names: list,
                 hidden_dim: int = 64):
        super().__init__()
        self.label_names = label_names
        self.heads = nn.ModuleDict()
        for name in label_names:
            self.heads[name] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, eeg_z: torch.Tensor,
                labels: dict) -> torch.Tensor:
        """
        eeg_z  : (B, D) EEG embeddings (pre-projection, from encoder)
        labels : dict mapping label_name -> (B,) tensor of 0/1/-1

        Returns scalar loss averaged over all heads with valid labels.
        """
        total_loss = torch.tensor(0.0, device=eeg_z.device)
        n_active = 0

        for name in self.label_names:
            if name not in labels:
                continue
            y = labels[name]  # (B,)
            valid = y != -1
            if valid.sum() == 0:
                continue

            logits = self.heads[name](eeg_z[valid]).squeeze(-1)  # (n_valid,)
            target = y[valid].float()
            loss = F.binary_cross_entropy_with_logits(logits, target)
            total_loss = total_loss + loss
            n_active += 1

        if n_active == 0:
            return total_loss  # zero -- no valid labels in this batch
        return total_loss / n_active


if __name__ == "__main__":
    print("[losses] Running sanity checks...")
    torch.manual_seed(42)

    B, D = 8, 64
    eeg_z = F.normalize(torch.randn(B, D), dim=-1)
    text_z = F.normalize(torch.randn(B, D), dim=-1)
    logit_scale = torch.tensor(14.0)  # ~ 1/0.07
    bias = torch.tensor(0.0)

    # 1. Sigmoid loss
    loss_sig = sigmoid_contrastive_loss(eeg_z, text_z, logit_scale, bias)
    print(f"[losses] SigLIP sigmoid loss (random): {loss_sig.item():.4f}")
    assert loss_sig.item() > 0, "Loss should be positive"

    # 2. Soft-target loss
    loss_soft = soft_target_contrastive_loss(eeg_z, text_z, logit_scale)
    print(f"[losses] Soft-target contrastive loss (random): {loss_soft.item():.4f}")
    assert loss_soft.item() > 0, "Loss should be positive"

    # 3. Auxiliary head loss
    aux = AuxiliaryHeadLoss(D, ["slowing", "epileptiform", "focal_left"])
    aux_labels = {
        "slowing": torch.tensor([1, 0, -1, 1, 0, 0, 1, -1]),
        "epileptiform": torch.tensor([0, 0, 1, -1, 0, 1, 0, 1]),
        "focal_left": torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1]),  # all unknown
    }
    loss_aux = aux(eeg_z, aux_labels)
    print(f"[losses] Auxiliary head loss (random): {loss_aux.item():.4f}")

    # 4. Gradient check
    eeg_z.requires_grad_(True)
    loss = sigmoid_contrastive_loss(eeg_z, text_z, logit_scale, bias)
    loss.backward()
    assert eeg_z.grad is not None, "Gradients should flow"
    print(f"[losses] Gradient norm: {eeg_z.grad.norm().item():.6f}")

    print("[losses] All checks passed. OK.")
