@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  echo Put this launcher beside the release files.
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  echo Run download-qwen38.bat first, or drag a qwen3_8_27b.ninfer file onto this launcher.
  exit /b 1
)

echo Starting Qwen3.8-27B on Dual RTX 3060 12GB (Tensor Split 50:50) at http://127.0.0.1:8080/v1
echo Profile: up to eight requests, 8K context, MTP3, ReplaySSM, Dual-GPU
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --devices 0,1 --tensor-split 0.5,0.5 --max-context 8192 --kv-capacity 16384 --max-concurrency 8 --max-pending-requests 32 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
