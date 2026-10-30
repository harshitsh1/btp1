@echo off
setlocal
echo ========================================================
echo Generating and Saving Spectrograms for TUAB EDF Files
echo Directing 100%% of caches, outputs, and temp to E: drive
echo ========================================================

set E_CACHE=E:\eeg_clip\.cache
set TORCH_HOME=%E_CACHE%\torch
set HF_HOME=%E_CACHE%\huggingface
set MPLCONFIGDIR=%E_CACHE%\matplotlib
set TEMP=%E_CACHE%\temp
set TMP=%E_CACHE%\temp
set TMPDIR=%E_CACHE%\temp
set CUDA_CACHE_PATH=%E_CACHE%\nvidia
set PYTHONPYCACHEPREFIX=%E_CACHE%\pycache

python save_spectrograms.py %*
endlocal
