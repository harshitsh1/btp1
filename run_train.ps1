# Run Multimodal EEG-CLIP Training with safe flags and full E: drive storage isolation
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting Multimodal EEG-CLIP Training (EEG + Text + Vision)" -ForegroundColor Green
Write-Host "Directing 100% of caches, checkpoints, logs, and temp to E: drive" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan

$E_CACHE = "E:\eeg_clip\.cache"
$env:TORCH_HOME = "$E_CACHE\torch"
$env:HF_HOME = "$E_CACHE\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "$E_CACHE\huggingface\hub"
$env:TRANSFORMERS_CACHE = "$E_CACHE\huggingface\hub"
$env:HF_DATASETS_CACHE = "$E_CACHE\huggingface\datasets"
$env:HF_METRICS_CACHE = "$E_CACHE\huggingface\metrics"
$env:HF_MODULES_CACHE = "$E_CACHE\huggingface\modules"
$env:MPLCONFIGDIR = "$E_CACHE\matplotlib"
$env:TEMP = "$E_CACHE\temp"
$env:TMP = "$E_CACHE\temp"
$env:TMPDIR = "$E_CACHE\temp"
$env:CUDA_CACHE_PATH = "$E_CACHE\nvidia"
$env:PYTHONPYCACHEPREFIX = "$E_CACHE\pycache"

New-Item -ItemType Directory -Force -Path "$E_CACHE\temp", "$E_CACHE\nvidia", "$E_CACHE\pycache" | Out-Null

python train.py --epochs 20 --batch_size 64 --use_vision --vision_encoder resnet18 --vision_image_mode topomap --freeze_text_layers 10 @args
