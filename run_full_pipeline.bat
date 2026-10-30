@echo off
setlocal
echo ========================================================
echo Running Full End-to-End Multimodal EEG-CLIP Pipeline
echo 1. Multimodal Pretraining (20 epochs on full TUAB corpus)
echo 2. Comprehensive Clinical Evaluation (Probes, Zero-Shot, CoOp, Few-Shot)
echo 3. Comparative Benchmark (Multimodal EEG-CLIP vs EEG-Only Baselines)
echo 4. Spectrogram and Topomap Generation
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

if /i "%1"=="--stage" (
    if "%2"=="2" goto step2
    if "%2"=="3" goto step3
    if "%2"=="4" goto step4
)
if /i "%1"=="--from_stage" (
    if "%2"=="2" goto step2
    if "%2"=="3" goto step3
    if "%2"=="4" goto step4
)
if /i "%1"=="--stage2" goto step2
if /i "%1"=="--from_stage_2" goto step2
if /i "%1"=="--skip_train" goto step2

:step1
echo.
echo [Step 1/4] Starting Full Multimodal Pretraining (High Throughput: Batch 128, 8 epochs)...
python train.py --epochs 8 --batch_size 128 --use_vision --vision_encoder resnet18 --vision_image_mode topomap --freeze_text_layers 10 %*
if errorlevel 1 (
    echo [ERROR] Step 1 Training failed.
    exit /b 1
)

:step2
echo.
echo [Step 2/4] Starting Comprehensive Clinical Evaluation...
python evaluate.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender medication --skip_baselines
if errorlevel 1 (
    echo [ERROR] Step 2 Evaluation failed.
    exit /b 1
)

:step3
echo.
echo [Step 3/4] Running Comparative Benchmark (Multimodal vs Baselines)...
python compare_eeg_vs_text.py --checkpoint outputs/checkpoints/eeg_clip_final.pt --tasks pathological age gender --epochs 10
if errorlevel 1 (
    echo [ERROR] Step 3 Comparison failed.
    exit /b 1
)

:step4
echo.
echo [Step 4/4] Generating Spectrogram Gallery...
python save_spectrograms.py --split eval
if errorlevel 1 (
    echo [ERROR] Step 4 Spectrogram generation failed.
    exit /b 1
)

echo.
echo ========================================================
echo FULL END-TO-END PIPELINE COMPLETED SUCCESSFULLY!
echo Results and reports saved to E:\eeg_clip\outputs
echo ========================================================
endlocal
