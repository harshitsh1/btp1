"""
models/prompt_learner.py
=========================
WHY
---
The original EEG-CLIP uses two fixed, hand-written text prompts per task
for zero-shot classification (e.g. "This is a normal recording" vs "This
is an abnormal recording"). These prompts are unoptimized and may not
align well with what the model actually learned during contrastive
pretraining. CoOp (Zhou et al., 2022) showed that learning continuous
prompt embeddings instead of using discrete text gives large zero-shot
boosts in CLIP-like models.

This module implements:

  1. **CoOp-style prompt learning**: learn M continuous token embeddings
     [v₁][v₂]...[vₘ] that are prepended to the class-name token, then
     passed through the frozen text encoder. The prompt vectors are the
     ONLY trainable parameters -- the text encoder stays frozen.

  2. **Unified prompt** (class-agnostic prefix) and **class-specific**
     variants:
       - Unified: all classes share the same [v₁]...[vₘ] prefix
       - Class-specific: each class has its own prefix

Both approaches are implemented via `PromptLearner`, which wraps the
existing text encoder and replaces its tokenize-then-encode pipeline with
a direct embedding injection.

HOW
---
    from models.prompt_learner import PromptLearner

    # Wrap a frozen text encoder
    prompt_learner = PromptLearner(
        text_encoder=model.text_encoder,
        class_names=["normal", "abnormal"],
        n_prompt_tokens=8,
        class_specific=False,
    )

    # Get prompted text embeddings (class_names order)
    text_z = prompt_learner()  # (n_classes, hidden_size)

Run standalone for a sanity check:
    python models/prompt_learner.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class PromptLearner(nn.Module):
    """
    CoOp-style learnable prompt embeddings for zero-shot EEG classification.

    Instead of "This is a normal recording" (a fixed string), we learn:
        [v₁][v₂]...[vₘ][CLASS_NAME]

    where [vᵢ] are learnable d-dimensional vectors in the text encoder's
    embedding space, and [CLASS_NAME] is the frozen embedding of the
    class name token(s).

    Parameters
    ----------
    text_encoder : TextEncoder
        The frozen text encoder (BERT etc.) -- its word embedding layer
        is used to embed class name tokens, and its transformer body
        processes the full prompt.
    class_names : list[str]
        Names of the classes (e.g. ["normal", "abnormal"]).
    n_prompt_tokens : int
        Number of learnable prompt tokens M (default 8, following CoOp).
    class_specific : bool
        If True, each class has its own set of M prompt tokens.
        If False, all classes share the same M tokens (unified prompt).
    prompt_init : str
        Initialization strategy: "random" or a template string like
        "This is a {} recording" where {} is replaced by class name.
    """

    def __init__(self, text_encoder: nn.Module, class_names: List[str],
                 n_prompt_tokens: int = 8, class_specific: bool = False,
                 prompt_init: str = "random"):
        super().__init__()
        self.text_encoder = text_encoder
        self.class_names = class_names
        self.n_prompt_tokens = n_prompt_tokens
        self.class_specific = class_specific
        self.n_classes = len(class_names)

        # Freeze the entire text encoder -- only prompt tokens are trainable
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Get the text encoder's embedding dimension
        self.embed_dim = text_encoder.hidden_size
        tokenizer = text_encoder.tokenizer

        # Tokenize and embed class names (frozen)
        self.class_token_ids = []
        self.class_token_lengths = []
        for name in class_names:
            tokens = tokenizer.encode(name, add_special_tokens=False)
            self.class_token_ids.append(tokens)
            self.class_token_lengths.append(len(tokens))

        # Get the word embedding layer from the transformer model
        self._word_embeddings = self._get_word_embeddings(text_encoder.model)

        # Initialize learnable prompt tokens
        if class_specific:
            # Each class gets its own prompt: (n_classes, M, embed_dim)
            self.prompt_embeddings = nn.Parameter(
                torch.empty(self.n_classes, n_prompt_tokens, self.embed_dim))
        else:
            # Shared prompt: (1, M, embed_dim) -- broadcast to all classes
            self.prompt_embeddings = nn.Parameter(
                torch.empty(1, n_prompt_tokens, self.embed_dim))

        self._init_prompts(prompt_init, tokenizer)

    def _get_word_embeddings(self, model):
        """Extract the word embedding layer from various HuggingFace models."""
        if hasattr(model, 'embeddings'):
            return model.embeddings.word_embeddings
        elif hasattr(model, 'shared'):
            return model.shared
        elif hasattr(model, 'wte'):
            return model.wte
        raise ValueError("Cannot find word embeddings in text encoder model. "
                         "Supported: BERT, GPT-2, T5-style models.")

    def _init_prompts(self, prompt_init: str, tokenizer):
        """Initialize prompt embeddings."""
        if prompt_init == "random":
            # Xavier uniform, same as typical embedding init
            nn.init.normal_(self.prompt_embeddings, std=0.02)
        else:
            # Initialize from a template string
            # e.g. "This is a {} recording" -> tokenize the non-{} parts
            parts = prompt_init.replace("{}", " PLACEHOLDER ").split()
            placeholder_idx = parts.index("PLACEHOLDER") if "PLACEHOLDER" in parts else -1
            template_words = [w for w in parts if w != "PLACEHOLDER"]

            # Tokenize template words and get their embeddings
            template_text = " ".join(template_words)
            token_ids = tokenizer.encode(template_text, add_special_tokens=False)

            with torch.no_grad():
                token_embeds = self._word_embeddings(
                    torch.tensor(token_ids))  # (n_tokens, embed_dim)

            # Fill prompt embeddings from template, cycling if needed
            n_template = token_embeds.shape[0]
            n_prompt = self.n_prompt_tokens
            if self.class_specific:
                for c in range(self.n_classes):
                    for i in range(n_prompt):
                        self.prompt_embeddings.data[c, i] = token_embeds[i % n_template]
            else:
                for i in range(n_prompt):
                    self.prompt_embeddings.data[0, i] = token_embeds[i % n_template]

    def forward(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Construct prompted embeddings for all classes and run them
        through the frozen text encoder.

        Returns
        -------
        class_embeddings : (n_classes, hidden_size) -- one embedding per class
        """
        if device is None:
            device = self.prompt_embeddings.device

        model = self.text_encoder.model
        tokenizer = self.text_encoder.tokenizer

        all_embeddings = []

        for c_idx in range(self.n_classes):
            # Get prompt tokens for this class
            if self.class_specific:
                prompt = self.prompt_embeddings[c_idx]  # (M, embed_dim)
            else:
                prompt = self.prompt_embeddings[0]      # (M, embed_dim)

            prompt = prompt.to(device)

            # Get frozen class name token embeddings
            class_ids = torch.tensor(
                self.class_token_ids[c_idx], device=device)
            with torch.no_grad():
                class_embeds = self._word_embeddings(class_ids)  # (n_class_tokens, embed_dim)

            # Get special token embeddings ([CLS], [SEP] for BERT)
            cls_id = tokenizer.cls_token_id or tokenizer.bos_token_id
            sep_id = tokenizer.sep_token_id or tokenizer.eos_token_id

            with torch.no_grad():
                cls_embed = self._word_embeddings(
                    torch.tensor([cls_id], device=device))  # (1, embed_dim)
                sep_embed = self._word_embeddings(
                    torch.tensor([sep_id], device=device))  # (1, embed_dim)

            # Construct full sequence: [CLS] [v1]...[vM] [class_tokens] [SEP]
            full_embeds = torch.cat([
                cls_embed,     # (1, D)
                prompt,        # (M, D) -- learnable
                class_embeds,  # (n_class_tokens, D)
                sep_embed,     # (1, D)
            ], dim=0).unsqueeze(0)  # (1, seq_len, D)

            seq_len = full_embeds.shape[1]
            attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)

            # Build position embeddings and token type embeddings
            # (BERT needs these on top of word embeddings)
            if hasattr(model, 'embeddings'):
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                token_type_ids = torch.zeros(1, seq_len, device=device, dtype=torch.long)

                # Add position and token_type embeddings to our custom word embeddings
                position_embeds = model.embeddings.position_embeddings(position_ids)
                token_type_embeds = model.embeddings.token_type_embeddings(token_type_ids)
                full_embeds = full_embeds + position_embeds + token_type_embeds
                full_embeds = model.embeddings.LayerNorm(full_embeds)
                full_embeds = model.embeddings.dropout(full_embeds)

                # Run through BERT encoder layers directly
                extended_mask = model.get_extended_attention_mask(
                    attention_mask, full_embeds.shape[:2], device)
                encoder_out = model.encoder(
                    full_embeds, attention_mask=extended_mask)
                hidden = encoder_out.last_hidden_state
            else:
                # Fallback for non-BERT models: use model's forward
                # with inputs_embeds (supported by most HF models)
                out = model(inputs_embeds=full_embeds,
                           attention_mask=attention_mask)
                hidden = out.last_hidden_state

            # Use CLS token output as the class embedding
            cls_output = hidden[:, 0, :]  # (1, hidden_size)
            all_embeddings.append(cls_output.squeeze(0))

        return torch.stack(all_embeddings, dim=0)  # (n_classes, hidden_size)


if __name__ == "__main__":
    print("[prompt_learner] Running sanity check with a dummy text encoder...\n")

    # Minimal dummy text encoder that mimics the real one's interface
    class _DummyTokenizer:
        cls_token_id = 101
        sep_token_id = 102
        bos_token_id = None
        eos_token_id = None
        def encode(self, text, add_special_tokens=False):
            # Fake tokenization: one token per word, IDs in [200, 300)
            words = text.split()
            return [200 + i for i in range(len(words))]

    class _DummyEmbeddings(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.word_embeddings = nn.Embedding(500, hidden_size)
            self.position_embeddings = nn.Embedding(512, hidden_size)
            self.token_type_embeddings = nn.Embedding(2, hidden_size)
            self.LayerNorm = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(0.0)

    class _DummyEncoder(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)

        def forward(self, x, **kwargs):
            out = self.linear(x)
            return type('Out', (), {'last_hidden_state': out})()

    class _DummyModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.embeddings = _DummyEmbeddings(hidden_size)
            self.encoder = _DummyEncoder(hidden_size)
            self.config = type('C', (), {'hidden_size': hidden_size})()

        def get_extended_attention_mask(self, mask, shape, device):
            return mask.unsqueeze(1).unsqueeze(2).float()

    class _DummyTextEncoder(nn.Module):
        def __init__(self, hidden_size=64):
            super().__init__()
            self.hidden_size = hidden_size
            self.model = _DummyModel(hidden_size)
            self.tokenizer = _DummyTokenizer()

    torch.manual_seed(42)
    text_enc = _DummyTextEncoder(hidden_size=64)

    learner = PromptLearner(
        text_encoder=text_enc,
        class_names=["normal", "abnormal"],
        n_prompt_tokens=8,
        class_specific=False,
        prompt_init="random",
    )

    n_trainable = sum(p.numel() for p in learner.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in learner.parameters())
    print(f"[prompt_learner] Trainable params: {n_trainable:,} / {n_total:,} total")

    with torch.no_grad():
        embeddings = learner()
    print(f"[prompt_learner] Output shape: {tuple(embeddings.shape)} "
          f"(expected (2, 64))")
    assert embeddings.shape == (2, 64), f"Shape mismatch: {embeddings.shape}"

    # Gradient check: only prompt_embeddings should have grad
    embeddings = learner()
    embeddings.sum().backward()
    assert learner.prompt_embeddings.grad is not None, "Prompt embeddings should have grad"
    for p in text_enc.parameters():
        if p.grad is not None:
            assert (p.grad == 0).all() or not p.requires_grad, \
                "Text encoder params should be frozen"

    print("[prompt_learner] All checks passed. OK.")
