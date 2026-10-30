@echo off
setlocal
echo ========================================================
echo Starting Multimodal EEG-CLIP Evaluation
echo Directing 100%% of caches, checkpoints, logs, and temp to E: drive
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

python evaluate.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender medication --skip_baselines %*
endlocal
