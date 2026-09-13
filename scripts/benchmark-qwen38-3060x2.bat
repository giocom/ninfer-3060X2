@echo off
setlocal
set "ROOT=%~dp0"
set "SERVER=%ROOT%ninfer-serve.exe"
set "MODEL=%~1"
if "%MODEL%"=="" set "MODEL=%ROOT%models\qwen3_8_27b.ninfer"

if not exist "%SERVER%" (
  echo Missing %SERVER%
  exit /b 1
)
if not exist "%MODEL%" (
  echo Missing model: %MODEL%
  exit /b 1
)

echo ========================================================
echo  NInfer Dual RTX 3060 12GB (Tensor Split 50:50) Benchmark
echo ========================================================
echo Running C1 131K Long-Context Benchmark...
"%SERVER%" "%MODEL%" --host 127.0.0.1 --port 8080 --devices 0,1 --tensor-split 0.5,0.5 --max-context 131072 --kv-capacity 131072 --max-concurrency 1 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
