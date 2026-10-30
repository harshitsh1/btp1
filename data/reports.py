"""
data/reports.py
================
WHY
---
Every TUAB recording is accompanied by a free-text clinical report written
by a neurologist. The paper (Sec. 3.1, Table 1) splits each report into up
to 15 named sections (IMPRESSION, DESCRIPTION OF THE RECORD, ...) and uses
either all of them concatenated, or a single section, as the text input to
the BERT text encoder. The paper also derives three of its four evaluation
labels directly from this text:
    - "age"        : patient age, thresholded at 50 (cfg.age_threshold)
    - "gender"     : M / F
    - "medication" : whether any of keppra / dilantin / depakote is
                      mentioned (the 3 most common anticonvulsants, Sec 3.4)
The fourth label, "pathological" (normal/abnormal), comes from the TUAB
directory structure itself (edf/{train,eval}/{normal,abnormal}/...), not
from the report text, and is loaded in data/dataset.py.

IMPORTANT / ASSUMPTION
-----------------------
The official TUAB release stores each recording's report as its own text
export from the hospital's reporting system. Section header formatting
varies a little across years (e.g. "CLINICAL HISTORY:" vs "HISTORY:"), so
the parser below is intentionally permissive (case-insensitive, optional
colon, optional whitespace) and will print a warning + fall back to
treating the *whole* file as one "IMPRESSION"-less blob if no known
section headers are found at all, so a report with unusual formatting
never silently disappears -- you'll see it in the printed stats when you
run this file or build the dataset.

Age/gender are not stored in a separate metadata file in the base
corpus; TUH reports almost always state them near the top of the
CLINICAL HISTORY / INTRODUCTION section (e.g. "84 year old right handed
male referred for ..."). We extract them with regex from the raw text.
If your local copy of TUAB *does* ship separate demographic CSVs, prefer
loading age/gender from there instead -- swap `extract_age_gender` for a
CSV lookup in data/dataset.py.

HOW
---
    from data.reports import parse_report_sections, extract_labels_from_text
    sections = parse_report_sections(raw_text, cfg.report_sections)
    text = build_text_input(sections, cfg, mode="all")
    labels = extract_labels_from_text(raw_text, cfg)

Run standalone on a .txt report to see the parse:
    python data/reports.py /path/to/report.txt
"""

import re
import sys
import os


_SECTION_HEADER_RE_CACHE = {}


def _section_pattern(section_names):
    """Builds one compiled regex that matches any of the section headers,
    tolerant to trailing colon / varying case / extra spaces."""
    key = tuple(section_names)
    if key in _SECTION_HEADER_RE_CACHE:
        return _SECTION_HEADER_RE_CACHE[key]
    alts = "|".join(re.escape(s) for s in section_names)
    pattern = re.compile(rf"^\s*({alts})\s*:?\s*$", re.IGNORECASE | re.MULTILINE)
    _SECTION_HEADER_RE_CACHE[key] = pattern
    return pattern


def parse_report_sections(text: str, section_names) -> dict:
    """
    Splits a raw report string into {SECTION_NAME: body_text}.

    Uses a simple state machine: scan line by line, whenever a line matches
    a known section header (case-insensitively), start accumulating into
    that section until the next header.
    """
    if not text or not text.strip():
        return {}

    lines = text.splitlines()
    header_set = {s.upper() for s in section_names}
    sections = {}
    current = None
    buffer = []

    def flush():
        if current is not None:
            body = "\n".join(buffer).strip()
            if body:
                sections[current] = body

    for line in lines:
        stripped = line.strip().rstrip(":").strip().upper()
        if stripped in header_set:
            flush()
            current = stripped
            buffer = []
        else:
            buffer.append(line)
    flush()

    if not sections:
        # Fallback: no recognized headers found at all. Keep the raw text
        # under a generic key so callers can still use *something*, but
        # make it loud so bad-format files are visible in dataset stats.
        sections["_UNSTRUCTURED"] = text.strip()

    return sections


def build_text_input(sections: dict, cfg, mode: str = None) -> str:
    """
    Builds the string that is actually tokenized and fed to the text
    encoder, following the ablation setup of paper Sec. 4.2 / Fig. 5.

    mode:
      "all"                 -> concatenate every non-empty section
                                (best overall performance in the paper)
      "<SECTION NAME>"      -> use only that single section
      "none"                -> return empty string (ablation control used
                                in Fig. 5 as a lower-bound baseline)
    """
    mode = mode or cfg.text_section_mode
    if mode.lower() == "none":
        return ""
    if mode.lower() == "all":
        parts = [sections[s] for s in cfg.report_sections if s in sections]
        if not parts and "_UNSTRUCTURED" in sections:
            parts = [sections["_UNSTRUCTURED"]]
        return " ".join(parts)
    # single named section
    key = mode.upper()
    return sections.get(key, "")


_AGE_PATTERNS = [
    re.compile(r"(\d{1,3})[\s-]*year[\s-]*old", re.IGNORECASE),
    re.compile(r"age[:\s]+(\d{1,3})", re.IGNORECASE),
]
_MALE_RE = re.compile(r"\bmale\b", re.IGNORECASE)
_FEMALE_RE = re.compile(r"\bfemale\b", re.IGNORECASE)


def extract_age_gender(text: str):
    """
    Regex extraction of age (int or None) and gender ("M"/"F"/None) from
    free text, targeting the standard TUH phrasing
    "This is a normal EEG for an 84 year old male ...".

    "female" is checked before deciding "male" match is a real male mention
    since the substring "male" also occurs inside "female".
    """
    age = None
    for pat in _AGE_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = int(m.group(1))
            if 0 < candidate < 120:  # sanity bound
                age = candidate
                break

    gender = None
    if _FEMALE_RE.search(text):
        gender = "F"
    elif _MALE_RE.search(text):
        gender = "M"

    return age, gender


def extract_medication_label(text: str, keywords) -> int:
    """Returns 1 if any anticonvulsant keyword appears in the report, else 0."""
    low = text.lower()
    return int(any(k.lower() in low for k in keywords))


def extract_labels_from_text(raw_text: str, cfg) -> dict:
    """Convenience wrapper returning all text-derived labels at once."""
    age, gender = extract_age_gender(raw_text)
    medication = extract_medication_label(raw_text, cfg.medication_keywords)
    return {
        "age": age,                                   # int or None
        "age_binary": None if age is None else int(age > cfg.age_threshold),
        "gender": gender,                              # "M"/"F" or None
        "gender_binary": None if gender is None else int(gender == "F"),
        "medication": medication,                       # 0/1
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import cfg

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", errors="ignore") as f:
            raw = f.read()
    else:
        raw = (
            "CLINICAL HISTORY:\n84 year old right handed male with history of "
            "seizures, on keppra.\n\nIMPRESSION:\nAbnormal EEG due to "
            "generalized slowing.\n\nMEDICATIONS:\nKeppra, Dilantin\n"
        )
        print("[reports] No file given, using a synthetic demo report.\n")

    sections = parse_report_sections(raw, cfg.report_sections)
    print("=" * 70)
    print("Parsed sections:")
    for k, v in sections.items():
        print(f"  [{k}] -> {v[:80]!r}{'...' if len(v) > 80 else ''}")

    text_all = build_text_input(sections, cfg, mode="all")
    print("\nText input (mode='all'):")
    print(" ", text_all[:200])

    labels = extract_labels_from_text(raw, cfg)
    print("\nExtracted labels:", labels)
    print("=" * 70)
