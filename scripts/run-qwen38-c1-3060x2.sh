#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="${ROOT}/ninfer-serve"
MODEL="${1:-${ROOT}/models/qwen3_8_27b.ninfer}"

if [[ ! -x "${SERVER}" ]]; then
  echo "Missing ${SERVER}"
  echo "Build or place ninfer-serve beside this launcher."
  exit 1
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "Missing model: ${MODEL}"
  echo "Run download-qwen38.sh first, or pass the path to a .ninfer model."
  exit 1
fi

echo "Starting Qwen3.8-27B on Dual RTX 3060 12GB (Tensor Split 50:50) at http://127.0.0.1:8080/v1"
echo "Profile: one request, 64K context, MTP3, ReplaySSM, Dual-GPU"
exec "${SERVER}" "${MODEL}" \
  --host 127.0.0.1 \
  --port 8080 \
  --devices 0,1 \
  --tensor-split 0.5,0.5 \
  --max-context 65536 \
  --kv-capacity 65536 \
  --max-concurrency 1 \
  --max-pending-requests 16 \
  --prefill-chunk 1024 \
  --kv-dtype int8 \
  --spec mtp \
  --draft-tokens 3 \
  --lm-head-draft
