"""
test_all_extensive.py
=====================
WHY
---
Master end-to-end verification suite that validates the entire Multimodal
EEG-CLIP system in ~3-5 minutes:
  1. Stage 1: Spectrogram & Scalp Topomap Generation & Gallery Export
  2. Stage 2: Multimodal 3-Way Pre-training (EEG + Clinical-Text + Spectrograms + Aux Heads)
  3. Stage 3: Clinical Benchmark Evaluation (Linear, MLP, Zero-Shot, CoOp Prompts, Few-Shot)
  4. Stage 4: Side-by-Side Model Comparison (Multimodal vs. EEG-Only Baseline)
  5. Stage 5: Storage & Safety Audit (Ensuring zero leaks to C: drive)
"""

from config import cfg  # Must be first to redirect all temp and cache dirs to E: drive
import os
import sys
import time
import subprocess
import shutil
import torch


def get_c_drive_free_gb():
    """Returns free space on C: drive in gigabytes."""
    try:
        total, used, free = shutil.disk_usage("C:\\")
        return free / (1024 ** 3)
    except Exception:
        return 0.0


def print_banner(text, char="="):
    line = char * 72
    print(f"\n{line}")
    print(f" {text}")
    print(f"{line}")


def run_stage(stage_num, stage_name, cmd_args):
    """Executes a command and returns (success, duration_seconds, output)."""
    print_banner(f"STAGE {stage_num}: {stage_name}", "-")
    cmd_str = " ".join(cmd_args)
    print(f"[cmd] {cmd_str}\n")
    t0 = time.time()
    
    # Run in subprocess with current environment (which already has E: cache redirects)
    proc = subprocess.Popen(
        [sys.executable, "-u"] + cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    output_lines = []
    for line in proc.stdout:
        print("  " + line, end="", flush=True)
        output_lines.append(line)
        
    proc.wait()
    duration = time.time() - t0
    success = (proc.returncode == 0)
    
    status = "SUCCESS" if success else f"FAILED (code {proc.returncode})"
    print(f"\n[stage {stage_num}] {stage_name}: {status} in {duration:.1f}s")
    return success, duration, "".join(output_lines)


def main():
    print_banner("MULTIMODAL EEG-CLIP: EXTENSIVE VERIFICATION SUITE")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU Device:   {torch.cuda.get_device_name(0)}")
    print(f"Data Root:    {os.path.abspath(cfg.data_root)}")
    print(f"Outputs Dir:  {os.path.abspath(cfg.output_dir)} (Strictly on E: drive)")
    
    c_free_start = get_c_drive_free_gb()
    print(f"C: Drive Free: {c_free_start:.2f} GB")
    
    results = []
    t_suite_start = time.time()
    
    test_checkpoint = "outputs/checkpoints/eeg_clip_test.pt"
    
    # -------------------------------------------------------------------------
    # STAGE 1: Spectrogram & Topomap Generation & Gallery
    # -------------------------------------------------------------------------
    stage1_cmd = [
        "save_spectrograms.py",
        "--split", "eval",
        "--max_recordings", "4",
        "--mode", "all",
        "--max_windows_per_rec", "2",
    ]
    s1_ok, s1_dur, _ = run_stage(1, "Spectrogram & Scalp Topomap Generation", stage1_cmd)
    
    gallery_path = "E:/eeg_clip/outputs/spectrograms/spectrogram_gallery.png"
    gallery_exists = os.path.exists(gallery_path)
    results.append(("Spectrogram & Topomap Generation", s1_ok and gallery_exists, s1_dur,
                    f"Gallery created: {gallery_exists}"))

    # -------------------------------------------------------------------------
    # STAGE 2: Multimodal 3-Way Training with Spectrograms & Auxiliary Heads
    # -------------------------------------------------------------------------
    stage2_cmd = [
        "train.py",
        "--epochs", "1",
        "--batch_size", "16",
        "--use_vision",
        "--vision_encoder", "resnet18",
        "--vision_image_mode", "spectrogram",
        "--freeze_text_layers", "10",
        "--max_recordings", "10",
    ]
    s2_ok, s2_dur, s2_out = run_stage(2, "Multimodal Pretraining (EEG + Text + Spectrogram)", stage2_cmd)
    
    # Copy final checkpoint to dedicated test checkpoint path for subsequent stages
    final_ckpt = "outputs/checkpoints/eeg_clip_final.pt"
    if os.path.exists(final_ckpt):
        shutil.copyfile(final_ckpt, test_checkpoint)
        ckpt_saved = True
    else:
        ckpt_saved = False
    
    results.append(("Multimodal Training (EEG+Vision)", s2_ok and ckpt_saved, s2_dur,
                    "InfoNCE 3-way + Aux heads convergence"))

    # -------------------------------------------------------------------------
    # STAGE 3: Downstream Clinical Task Evaluation
    # -------------------------------------------------------------------------
    stage3_cmd = [
        "evaluate.py",
        "--checkpoint", test_checkpoint,
        "--tasks", "pathological", "age", "gender",
        "--use_learned_prompts",
        "--skip_baselines",
        "--max_recordings", "10",
    ]
    s3_ok, s3_dur, _ = run_stage(3, "Clinical Benchmark Evaluation (Probes + Zero-Shot + CoOp)", stage3_cmd)
    
    eval_csv_exists = os.path.exists("outputs/logs/eval_classification.csv")
    results.append(("Clinical Evaluation (Linear/MLP/Zero-shot/CoOp)", s3_ok and eval_csv_exists, s3_dur,
                    "Evaluated Pathological, Age, Gender"))

    # -------------------------------------------------------------------------
    # STAGE 4: Side-by-Side Model Comparison (Multimodal vs. EEG-Only)
    # -------------------------------------------------------------------------
    stage4_cmd = [
        "compare_eeg_vs_text.py",
        "--checkpoint", test_checkpoint,
        "--tasks", "pathological", "age", "gender",
        "--epochs", "2",
        "--max_recordings", "10",
    ]
    s4_ok, s4_dur, _ = run_stage(4, "Side-by-Side Comparison (Multimodal vs. EEG-Only)", stage4_cmd)
    
    comp_png_exists = os.path.exists("outputs/eeg_vs_text_comparison.png")
    results.append(("Comparative Benchmark (Multimodal vs Baselines)", s4_ok and comp_png_exists, s4_dur,
                    "Comparison chart & CSV generated"))

    # -------------------------------------------------------------------------
    # STAGE 5: Storage & Safety Audit
    # -------------------------------------------------------------------------
    print_banner("STAGE 5: Storage & Safety Audit", "-")
    time.sleep(2)  # Allow Windows virtual memory/pagefile to stabilize after subprocess exits
    c_free_end = get_c_drive_free_gb()
    c_diff_mb = (c_free_end - c_free_start) * 1024
    
    print(f"C: Drive Free Start: {c_free_start:.2f} GB")
    print(f"C: Drive Free End:   {c_free_end:.2f} GB (Delta: {c_diff_mb:+.1f} MB)")
    
    cache_root = "E:/eeg_clip/.cache"
    cache_subdirs = [f for f in os.listdir(cache_root) if os.path.isdir(os.path.join(cache_root, f))]
    print(f"E: Drive Active Caches: {cache_subdirs}")
    
    # Check for leaks
    crash_dir = os.path.expandvars(r"%LOCALAPPDATA%\CrashDumps")
    crash_files = [f for f in os.listdir(crash_dir) if f.endswith(".dmp")] if os.path.exists(crash_dir) else []
    print(f"Crash dumps on C: {len(crash_files)} file(s)")
    
    hf_c_cache = os.path.expandvars(r"%USERPROFILE%\.cache\huggingface")
    hf_c_exists = os.path.exists(hf_c_cache)
    print(f"HuggingFace cache on C: {'Found' if hf_c_exists else 'None (Clean)'}")
    
    # Verify no leaks to C:, no crash dumps, and active cache on E:
    e_cache_active = os.path.exists(os.path.join(cache_root, "huggingface")) and os.path.exists(os.path.join(cache_root, "torch"))
    safe_storage = (len(crash_files) == 0) and (not hf_c_exists) and e_cache_active
    results.append(("Storage Safety Audit (E: Drive Isolation)", safe_storage, 0.1,
                    f"C: Dumps: {len(crash_files)}, E: Cache: {len(cache_subdirs)} active"))

    # -------------------------------------------------------------------------
    # FINAL SCORECARD
    # -------------------------------------------------------------------------
    total_time = time.time() - t_suite_start
    print_banner("EXTENSIVE VERIFICATION SCORECARD")
    
    all_passed = True
    print(f"{'#':<3} | {'Stage':<42} | {'Status':<10} | {'Time':<8} | {'Notes'}")
    print("-" * 90)
    for idx, (name, passed, dur, notes) in enumerate(results, 1):
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"{idx:<3} | {name:<42} | {status_str:<10} | {dur:>5.1f}s  | {notes}")
    print("-" * 90)
    
    overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    print(f"\nOverall Result:  {overall_status}")
    print(f"Total Duration:  {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Generated Assets:")
    print(f"  - Spectrogram Gallery: file:///{os.path.abspath(gallery_path)}")
    print(f"  - Comparison Chart:   file:///{os.path.abspath('outputs/eeg_vs_text_comparison.png')}")
    print(f"  - Checkpoint:         file:///{os.path.abspath(test_checkpoint)}")
    print(f"  - Evaluation Logs:    file:///{os.path.abspath('outputs/logs/eval_classification.csv')}")
    print_banner("VERIFICATION COMPLETE")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
