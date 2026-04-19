@echo off
chcp 65001 >nul
echo  正在启动SAM服务

set PYTHON=%CD%\SAM3mmgp_env\python.exe
set FF_PATH=%CD%\ffmpeg-8.1-full_build-shared\bin
set CONDA_LIB=%CD%\SAM3mmgp_env\Library\bin
set CU_PATH=%CD%\SAM3mmgp_env\Lib\site-packages\torch\lib
set SC_PATH=%CD%\SAM3mmgp_env\Scripts
set PATH=%FF_PATH%;%CONDA_LIB%;%CU_PATH%;%SC_PATH%;%PATH%
set HF_ENDPOINT=https://hf-mirror.com
set HF_HOME=%CD%\.huggingface
set TORCH_HOME=%CD%\.huggingface
set XFORMERS_FORCE_DISABLE_TRITON=1
set FFMPEG_PATH=%CD%\ffmpeg-8.1-full_build-shared\bin

%PYTHON% gradio_app.py

pause
