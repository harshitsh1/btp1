@echo off
setlocal
echo ========================================================
echo Running Post-Training Pipeline (Steps 2, 3, and 4)
echo Model Checkpoint: outputs/checkpoints/eeg_clip_final.pt
echo 1. Clinical Evaluation (Probes, Zero-Shot, CoOp Prompts)
echo 2. Comparative Benchmark (Multimodal vs Baselines)
echo 3. Spectrogram and Topomap Gallery Export
echo Directing 100%% of caches, logs, and temp to E: drive
echo ========================================================

set E_CACHE=E:\eeg_clip\.cache
set TORCH_HOME=%E_CACHE%\torch
set HF_HOME=%E_CACHE%\huggingface
set HUGGINGFACE_HUB_CACHE=%E_CACHE%\huggingface\hub
set TRANSFORMERS_CACHE=%E_CACHE%\huggingface\hub
set HF_DATASETS_CACHE=%E_CACHE%\huggingface\datasets
set HF_METRICS_CACHE=%E_CACHE%\huggingface\metrics
set HF_MODULES_CACHE=%E_CACHE%\huggingface\modules
set MPLCONFIGDIR=%E_CACHE%\matplotlib
set TEMP=%E_CACHE%\temp
set TMP=%E_CACHE%\temp
set TMPDIR=%E_CACHE%\temp
set CUDA_CACHE_PATH=%E_CACHE%\nvidia
set PYTHONPYCACHEPREFIX=%E_CACHE%\pycache

echo.
echo [Step 2/4] Starting Comprehensive Clinical Evaluation...
python evaluate.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender medication --skip_baselines %*
if errorlevel 1 (
    echo [ERROR] Step 2 Evaluation failed.
    exit /b 1
)

echo.
echo [Step 3/4] Running Comparative Benchmark (Multimodal vs Baselines)...
python compare_eeg_vs_text.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender --epochs 5 %*
if errorlevel 1 (
    echo [ERROR] Step 3 Comparison failed.
    exit /b 1
)

echo.
echo [Step 4/4] Generating Spectrogram Gallery...
python save_spectrograms.py --split eval
if errorlevel 1 (
    echo [ERROR] Step 4 Spectrogram generation failed.
    exit /b 1
)

echo.
echo ========================================================
echo POST-TRAINING PIPELINE COMPLETED SUCCESSFULLY!
echo Results saved to E:\eeg_clip\outputs
echo ========================================================
endlocal
