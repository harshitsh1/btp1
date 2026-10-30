# Fix NaN Labels for Age/Gender/Medication Tasks

## Problem

The TUAB v3.0.1 dataset you have contains **only `.edf` files — no `.txt` report files**. The current code tries to extract age, gender, and medication labels from report text, but since no reports exist, all these labels are `None` → `-1` → **NaN** during evaluation.

However, the **EDF file headers contain age and gender** directly:
```
aaaaaaaq F 01-JAN-0000 aaaaaaaq Age:26
```
- `F` = Female, `M` = Male
- `Age:26` = 26 years old

**Medication** labels cannot be extracted without report text — this task will be marked as unavailable.

## Proposed Changes

### Data Pipeline

#### [MODIFY] [dataset.py](file:///e:/eeg_clip/data/dataset.py)

- In `_build_index()`: when no report `.txt` is found, **extract age and gender from the EDF header** using MNE's `subject_info` (which already parses the EDF patient field during `load_and_preprocess_edf`).
- Add a new helper `_extract_labels_from_edf_header(edf_path, cfg)` that:
  1. Reads the first 256 bytes of the EDF file (the fixed-length patient header)
  2. Parses `Age:XX` with regex
  3. Parses gender from the `M`/`F` field in the standard EDF+ patient ID format
  4. Returns the same label dict format as `extract_labels_from_text`
- This is lightweight (no full EDF load needed — just reads 256 bytes)
- Medication remains `None` when no report exists (this is correct — the info simply isn't available)

#### [MODIFY] [dataset.py](file:///e:/eeg_clip/data/dataset.py) — cache invalidation

- **Delete old cache files** — the existing cache has `None` for all age/gender labels. Users must rebuild.
- Print a clear message telling the user the cache is being rebuilt.

---

### Evaluation Robustness

#### [MODIFY] [evaluate.py](file:///e:/eeg_clip/evaluate.py)

- In `run_few_shot()` → `train_task_specific_deep4()` call: **skip tasks with 0 valid samples** instead of crashing with `ValueError: num_samples=0`.
- The crash happens at line 158 when creating a `DataLoader` with an empty filtered dataset. Add a guard: if the filtered dataset has 0 items, return `float("nan")`.

---

## rsync Issue

The rsync error (`connection unexpectedly closed`) is a **network/SSH issue**, not a code bug. The server at `10.14.16.36` responds to ping (high latency: 81ms–1068ms) but SSH connection drops immediately. Common causes:

1. **Wrong password** — the password prompt appeared but auth failed
2. **SSH config on remote** — remote may block password auth, or require key-based auth  
3. **Firewall / timeout** — high-latency link (>1s) may cause SSH handshake timeout

Try these steps to debug:
```powershell
# 1. Test SSH connection directly first
ssh -v deepak@10.14.16.36

# 2. If that works, try rsync with SSH debug
rsync -avh --progress -e "ssh -v" /e/tuh_eeg_data/ deepak@10.14.16.36:~/path/

# 3. If password auth is failing, try generating & copying an SSH key
ssh-keygen -t ed25519
ssh-copy-id deepak@10.14.16.36
```

> [!IMPORTANT]
> The rsync issue is independent of the codebase — it requires debugging your SSH connection to the remote server. The fixes above are for the code issues.

## Verification Plan

### Automated Tests
```powershell
# Quick test: verify EDF header parsing works on a sample file
python -c "from data.dataset import _extract_labels_from_edf_header; from config import cfg; print(_extract_labels_from_edf_header(r'E:\tuh_eeg_data\v3.0.1\edf\train\abnormal\01_tcp_ar\aaaaaaaq_s004_t000.edf', cfg))"

# Full run: rebuild cache and evaluate
python evaluate.py
```

### Manual Verification
- Confirm dataset summary shows non-zero counts for "with age/gender labels"
- Confirm evaluation no longer shows NaN for age and gender tasks
- Confirm medication shows NaN (expected — no data source) without crashing
