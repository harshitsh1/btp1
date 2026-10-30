"""
models/text_encoder.py
=======================
WHY
---
The paper uses "pretrained text encoders based on the BERT architecture"
(Sec. 3.3) and studies several variants in Sec. 4.3.1 / Fig. 6:
  - bert-base-uncased        (general domain)
  - medicalai/ClinicalBERT   (clinical notes; best overall in the paper)
  - microsoft/BioGPT-Large-PubMedQA (biomedical; underperformed)
  - BAAI/bge-large-en-v1.5   (scientific papers; strong on pathology/age)

We wrap HuggingFace `AutoModel` + `AutoTokenizer` so switching between
these is a one-line config change (cfg.text_encoder_name), matching how
the paper produces Figure 6.

We use the [CLS] token's last-hidden-state as the sentence embedding for
encoder-only models (BERT-family). For encoder architectures without a
pooled [CLS] representation, we mean-pool the last hidden state instead
(handled automatically).

HOW
---
    from models.text_encoder import TextEncoder
    enc = TextEncoder("bert-base-uncased", embed_dim=... )
    z = enc(["This is a normal recording", "Left temporal sharp waves"])

Run standalone for a shape sanity check (downloads the pretrained weights
the first time -- needs internet access):
    python models/text_encoder.py
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Wraps a pretrained transformer as a sentence encoder producing one
    embedding per input string, at the model's native hidden size (e.g.
    768 for BERT-base). The projection down to the shared contrastive
    space happens separately in models/eeg_clip.py's ProjectionHead, so
    this module stays a clean "text -> raw embedding" function.
    """

    def __init__(self, model_name: str = "bert-base-uncased",
                 max_length: int = 256, freeze_n_layers: int = 0):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        print(f"[text_encoder] Loading pretrained text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.hidden_size = self.model.config.hidden_size
        print(f"[text_encoder] Loaded. Hidden size = {self.hidden_size}")

        if freeze_n_layers > 0:
            self._freeze_lower_layers(freeze_n_layers)

    def _freeze_lower_layers(self, n_layers: int):
        """Optionally freeze the first n encoder layers -- useful if you
        want to fine-tune only the top of a large model to save GPU
        memory. The paper fine-tunes the whole text encoder (Sec 3.3),
        so this is off (0) by default."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
                frozen += 1
                continue
            for i in range(n_layers):
                if f"layer.{i}." in name or f"h.{i}." in name:
                    param.requires_grad = False
                    frozen += 1
        print(f"[text_encoder] Froze {frozen} parameter tensors "
              f"(first {n_layers} layers + embeddings).")

    def forward(self, texts, device=None) -> torch.Tensor:
        """
        texts: list[str] (empty strings are fine -- tokenizer pads them)
        returns: (batch, hidden_size) float tensor
        """
        device = device or next(self.model.parameters()).device
        # Replace empty strings with a neutral placeholder token so the
        # tokenizer never receives a fully-empty sequence.
        safe_texts = [t if t.strip() else "[EMPTY]" for t in texts]
        enc = self.tokenizer(
            safe_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)

        out = self.model(**enc)
        last_hidden = out.last_hidden_state  # (B, L, H)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # Mean-pool over non-padding tokens.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts

        return pooled


if __name__ == "__main__":
    enc = TextEncoder("bert-base-uncased", max_length=64)
    texts = ["This is a normal recording", "Left temporal sharp waves at 6 Hz"]
    with torch.no_grad():
        z = enc(texts, device=torch.device("cpu"))
    print(f"[text_encoder] Output shape: {tuple(z.shape)} "
          f"(expected (2, {enc.hidden_size}))")
    assert z.shape == (2, enc.hidden_size)
    print("[text_encoder] OK.")
