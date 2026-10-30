"""
data/structured_labels.py
==========================
WHY
---
The original EEG-CLIP trains purely with noisy free-text contrastive
supervision. But clinical EEG reports contain structured clinical
information that can be extracted and used as auxiliary supervision
targets. This module parses reports into structured multi-label
attributes using rule-based regex extraction (fast, no LLM dependency):

  - slowing           : generalized or focal slowing mentioned
  - epileptiform      : spikes, sharp waves, epileptiform discharges
  - focal_left        : left-sided lateralization
  - focal_right       : right-sided lateralization
  - generalized       : generalized findings (not focal)
  - background_normal : normal background rhythm / alpha posterior dominant
  - delta             : delta activity mentioned
  - theta             : theta activity mentioned
  - alpha             : alpha activity mentioned
  - beta              : beta activity mentioned

Each label is 1 (present), 0 (absent), or -1 (could not be determined
-- e.g. section missing or empty report). The -1 sentinel is masked out
during auxiliary loss computation in models/losses.py.

An optional LLM-based extractor (`extract_structured_labels_llm`) is
provided for higher-quality extraction when an OpenAI-compatible API is
available. It uses a structured JSON schema prompt to get consistent
outputs. The rule-based extractor is the default and requires no API.

HOW
---
    from data.structured_labels import extract_structured_labels
    labels = extract_structured_labels(report_text)
    # labels: dict[str, int]  e.g. {"slowing": 1, "epileptiform": 0, ...}

Run standalone on a report string:
    python data/structured_labels.py
"""

import re
from typing import Dict, Optional


# ── Label definitions ────────────────────────────────────────────────

STRUCTURED_LABEL_NAMES = [
    "slowing",
    "epileptiform",
    "focal_left",
    "focal_right",
    "generalized",
    "background_normal",
    "delta",
    "theta",
    "alpha",
    "beta",
]


# ── Rule-based regex extraction ──────────────────────────────────────

# WHY regexes: they're fast (no API cost, no latency), deterministic,
# and interpretable. For clinical EEG reports from TUAB, the vocabulary
# is highly stereotyped -- neurologists use a small set of standard
# phrases. Regex precision is high when the vocabulary is constrained.

_SLOWING_PATTERNS = [
    re.compile(r"\bslow(ing|ed)\b", re.IGNORECASE),
    re.compile(r"\bdiffuse\s+slow", re.IGNORECASE),
    re.compile(r"\bgeneralized\s+slow", re.IGNORECASE),
    re.compile(r"\bfocal\s+slow", re.IGNORECASE),
    re.compile(r"\bintermittent\s+(rhythmic\s+)?delta", re.IGNORECASE),
]

_EPILEPTIFORM_PATTERNS = [
    re.compile(r"\bepileptiform\b", re.IGNORECASE),
    re.compile(r"\bsharp\s+(wave|transient|activit)", re.IGNORECASE),
    re.compile(r"\bspike(s|[-\s]and[-\s]wave)?\b", re.IGNORECASE),
    re.compile(r"\bseizure", re.IGNORECASE),
    re.compile(r"\bictal\b", re.IGNORECASE),
    re.compile(r"\bPLEDs?\b"),  # periodic lateralized epileptiform discharges
    re.compile(r"\bGPEDs?\b"),  # generalized periodic epileptiform discharges
    re.compile(r"\bperiodic\s+(lateralized\s+)?epileptiform", re.IGNORECASE),
]

_LEFT_PATTERNS = [
    re.compile(r"\bleft\s+(temporal|frontal|parietal|occipital|central|hemisphere)", re.IGNORECASE),
    re.compile(r"\bF7\b|\bT3\b|\bT5\b|\bFP1\b|\bF3\b|\bC3\b|\bP3\b|\bO1\b"),
    re.compile(r"\bleft[-\s]sided\b", re.IGNORECASE),
]

_RIGHT_PATTERNS = [
    re.compile(r"\bright\s+(temporal|frontal|parietal|occipital|central|hemisphere)", re.IGNORECASE),
    re.compile(r"\bF8\b|\bT4\b|\bT6\b|\bFP2\b|\bF4\b|\bC4\b|\bP4\b|\bO2\b"),
    re.compile(r"\bright[-\s]sided\b", re.IGNORECASE),
]

_GENERALIZED_PATTERNS = [
    re.compile(r"\bgeneralized\b", re.IGNORECASE),
    re.compile(r"\bdiffuse\b", re.IGNORECASE),
    re.compile(r"\bbilateral(ly)?\b", re.IGNORECASE),
]

_BACKGROUND_NORMAL_PATTERNS = [
    re.compile(r"\bnormal\s+(awake\s+)?(EEG|background|recording)\b", re.IGNORECASE),
    re.compile(r"\bposterior\s+dominant\s+rhythm\b", re.IGNORECASE),
    re.compile(r"\bPDR\b"),
    re.compile(r"\balpha\s+rhythm\s+(is\s+)?present\b", re.IGNORECASE),
    re.compile(r"\bwell[- ]organized\s+background\b", re.IGNORECASE),
]

# Negation check: if the match is preceded by "no ", "no evidence of ",
# "without ", "absence of " within 40 characters, flip the label.
_NEGATION_RE = re.compile(
    r"\b(no|not|without|absence\s+of|no\s+evidence\s+of|no\s+definite)\s*$",
    re.IGNORECASE
)

_FREQ_BAND_PATTERNS = {
    "delta": re.compile(r"\bdelta\b", re.IGNORECASE),
    "theta": re.compile(r"\btheta\b", re.IGNORECASE),
    "alpha": re.compile(r"\balpha\b", re.IGNORECASE),
    "beta":  re.compile(r"\bbeta\b", re.IGNORECASE),
}


def _has_pattern(text: str, patterns: list, check_negation: bool = True) -> Optional[bool]:
    """Search for any pattern in text. Returns True/False, or None if
    the report text is empty / too short to be meaningful."""
    if not text or len(text.strip()) < 10:
        return None

    for pat in patterns:
        match = pat.search(text)
        if match:
            if check_negation:
                # Check the 40 chars before the match for negation words
                start = max(0, match.start() - 40)
                prefix = text[start:match.start()]
                if _NEGATION_RE.search(prefix):
                    return False
            return True
    return False


def extract_structured_labels(text: str) -> Dict[str, int]:
    """
    Extract structured multi-label attributes from a clinical EEG report
    using rule-based regex matching.

    Returns a dict mapping label names to 1 (present), 0 (absent),
    or -1 (undetermined -- report too short or missing).

    WHY -1 instead of 0 for missing: absence of evidence is not evidence
    of absence. If the report doesn't mention slowing, it might mean
    slowing is absent, OR it might mean the section was truncated /
    missing / uses unusual phrasing. The -1 sentinel lets the auxiliary
    loss skip these samples rather than training on potentially wrong
    labels.
    """
    if not text or len(text.strip()) < 10:
        return {name: -1 for name in STRUCTURED_LABEL_NAMES}

    results = {}

    # ── Binary clinical findings ──
    slowing = _has_pattern(text, _SLOWING_PATTERNS)
    results["slowing"] = -1 if slowing is None else int(slowing)

    epileptiform = _has_pattern(text, _EPILEPTIFORM_PATTERNS)
    results["epileptiform"] = -1 if epileptiform is None else int(epileptiform)

    # ── Lateralization ──
    focal_left = _has_pattern(text, _LEFT_PATTERNS, check_negation=False)
    results["focal_left"] = -1 if focal_left is None else int(focal_left)

    focal_right = _has_pattern(text, _RIGHT_PATTERNS, check_negation=False)
    results["focal_right"] = -1 if focal_right is None else int(focal_right)

    generalized = _has_pattern(text, _GENERALIZED_PATTERNS, check_negation=False)
    results["generalized"] = -1 if generalized is None else int(generalized)

    # ── Background rhythm ──
    bg_normal = _has_pattern(text, _BACKGROUND_NORMAL_PATTERNS)
    results["background_normal"] = -1 if bg_normal is None else int(bg_normal)

    # ── Frequency bands ──
    for band, pat in _FREQ_BAND_PATTERNS.items():
        match = pat.search(text) if text else None
        results[band] = int(bool(match)) if text and len(text.strip()) >= 10 else -1

    return results


# ── Optional LLM-based extraction ────────────────────────────────────

LLM_PROMPT_TEMPLATE = """You are a clinical EEG report parser. Extract the following binary attributes from the EEG report below. Return ONLY a JSON object with these keys, each mapping to 0 (absent/not mentioned) or 1 (present/mentioned):

- slowing: generalized or focal slowing
- epileptiform: epileptiform discharges, sharp waves, spikes, seizures
- focal_left: left-hemisphere lateralized findings
- focal_right: right-hemisphere lateralized findings
- generalized: generalized (bilateral/diffuse) findings
- background_normal: normal background rhythm / posterior dominant rhythm present
- delta: delta frequency band activity
- theta: theta frequency band activity
- alpha: alpha frequency band activity
- beta: beta frequency band activity

EEG Report:
{report_text}

JSON output:"""


def extract_structured_labels_llm(text: str, client=None,
                                   model: str = "gpt-4o-mini") -> Dict[str, int]:
    """
    Extract structured labels using an LLM API call.

    WHY: LLMs handle negation, coreference, and unusual phrasing far
    better than regex. The tradeoff is cost and latency -- use this for
    offline label generation (run once, cache results), not at training
    time.

    Requires `openai` package and OPENAI_API_KEY environment variable,
    or pass a pre-configured client.
    """
    import json as json_mod

    if client is None:
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            print("[structured_labels] openai package not installed. "
                  "Falling back to rule-based extraction.")
            return extract_structured_labels(text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise clinical EEG report parser. Output only valid JSON."},
                {"role": "user", "content": LLM_PROMPT_TEMPLATE.format(report_text=text)},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        parsed = json_mod.loads(content)
        # Validate and normalize
        result = {}
        for name in STRUCTURED_LABEL_NAMES:
            val = parsed.get(name)
            if val is None or val not in (0, 1):
                result[name] = -1
            else:
                result[name] = int(val)
        return result
    except Exception as e:
        print(f"[structured_labels] LLM extraction failed: {e}. "
              f"Falling back to rule-based.")
        return extract_structured_labels(text)


if __name__ == "__main__":
    print("[structured_labels] Running extraction demo...\n")

    demo_reports = [
        # 1. Abnormal with slowing and epileptiform
        ("Abnormal EEG with focal features",
         "IMPRESSION: Abnormal EEG due to left temporal sharp waves "
         "and focal slowing in the left temporal region. "
         "CLINICAL HISTORY: 67 year old male with seizures. "
         "DESCRIPTION OF THE RECORD: The background shows theta "
         "slowing with intermittent delta activity. Alpha rhythm "
         "is poorly organized."),

        # 2. Normal recording
        ("Normal EEG",
         "IMPRESSION: Normal awake EEG. "
         "DESCRIPTION OF THE RECORD: The posterior dominant rhythm "
         "consists of well-organized 10 Hz alpha activity. "
         "No epileptiform discharges were seen. No focal slowing."),

        # 3. Generalized abnormality
        ("Generalized slowing",
         "IMPRESSION: Abnormal EEG due to diffuse generalized slowing. "
         "DESCRIPTION OF THE RECORD: The background is characterized by "
         "generalized delta and theta activity bilaterally."),

        # 4. Empty/minimal report
        ("Empty report", ""),
    ]

    for title, text in demo_reports:
        labels = extract_structured_labels(text)
        print(f"--- {title} ---")
        for k, v in labels.items():
            symbol = {1: "[+]", 0: "[-]", -1: "[?]"}[v]
            print(f"  {k:20s}: {symbol} ({v})")
        print()

    print("[structured_labels] OK.")
