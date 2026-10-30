@echo off
setlocal
echo ========================================================
echo Starting Multimodal EEG-CLIP Training (EEG + Text + Vision)
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

if not exist "%E_CACHE%\temp" mkdir "%E_CACHE%\temp"
if not exist "%E_CACHE%\nvidia" mkdir "%E_CACHE%\nvidia"
if not exist "%E_CACHE%\pycache" mkdir "%E_CACHE%\pycache"

python train.py --epochs 5 --batch_size 128 --use_vision --vision_encoder resnet18 --vision_image_mode topomap --freeze_text_layers 10 %*
endlocal
