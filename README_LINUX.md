# NInfer for Linux (Ubuntu / Debian / Docker)
## Dual NVIDIA GeForce RTX 3060 12GB (24GB VRAM) & RTX 3090

This guide explains how to build, configure, and run **NInfer** on **Linux (Ubuntu 22.04 / 24.04, Debian, Fedora, Bazzite)** for **Dual RTX 3060 12GB (Tensor Split 50:50)** with **131K+ token context support**.

---

## 1. System Prerequisites

- **OS**: Ubuntu 22.04 / 24.04 LTS, Debian 12, or any modern Linux distribution
- **GPU**: 2x NVIDIA GeForce RTX 3060 12GB (or 1x RTX 3090 24GB)
- **NVIDIA Driver**: Version >= 570
- **CUDA Toolkit**: Version >= 12.8

Check GPUs:
```bash
nvidia-smi
```

---

## 2. Option A: Docker Setup (Recommended)

### 1) Build Image
```bash
docker build --tag ninfer:3060x2 .
```

### 2) Download Model
```bash
mkdir -p models
bash scripts/download-qwen38.sh
```

### 3) Run Dual 3060 Server (131K Context, Tensor Split 50:50)
```bash
docker run --rm --gpus all \
  --publish 8080:8080 \
  --volume "$PWD/models:/workspace/models:ro" \
  ninfer:3060x2 \
  ninfer-serve models/qwen3_8_27b.ninfer \
  --host 0.0.0.0 --port 8080 \
  --devices 0,1 \
  --tensor-split 0.5,0.5 \
  --max-context 131072 --kv-capacity 131072 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

---

## 3. Option B: Native Ubuntu 24.04 / 22.04 Build

### 1) Install Build Dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential gcc-13 g++-13 cmake ninja-build pkg-config \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev
```

### 2) Compile NInfer (SM86 Architecture)
```bash
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDAHOSTCXX=/usr/bin/g++-13

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DNINFER_BUILD_APPS=ON

cmake --build build --parallel
```

Compiled binaries will be located at:
- `build/apps/ninfer-serve` (HTTP Server for OpenAI/Anthropic APIs)
- `build/apps/ninfer` (Interactive CLI Tool)

---

## 4. Launch Scripts for Dual RTX 3060 12GB

| Script | Profile | Max Context | Description |
|---|---|---|---|
| `bash scripts/run-qwen38-3060x2-131k.sh` | Ultra Long Context | **131,072 tokens (131K)** | 50:50 Tensor Split, INT8 KV Cache, MTP3 Speculative |
| `bash scripts/run-qwen38-c1-3060x2.sh` | Interactive Single User | **65,536 tokens (64K)** | Lowest latency, single request |
| `bash scripts/run-qwen38-c8-3060x2.sh` | Multi-User Serving | **8,192 tokens (8K)** | Up to 8 concurrent streams |

---

## 5. Dual RTX 3060 12GB VRAM Architecture

| Component | GPU 0 (12GB) | GPU 1 (12GB) | Total (24GB) |
|---|---:|---:|---:|
| Model Weights (Qwen3.8-27B) | ~7.4 GiB | ~7.4 GiB | ~14.8 GiB |
| KV Cache (131K INT8) | ~3.8 GiB | ~3.8 GiB | ~7.6 GiB |
| Workspace & CUDA Graphs | ~0.5 GiB | ~0.5 GiB | ~1.0 GiB |
| **Total Allocated VRAM** | **~11.7 GiB** | **~11.7 GiB** | **~23.4 GiB** |

---

## 6. CLI Inference Example

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Write a Python script for fast asynchronous HTTP requests." \
  --devices 0,1 \
  --tensor-split 0.5,0.5 \
  --max-context 131072 \
  --kv-capacity 131072 \
  --kv-dtype int8 \
  --spec mtp \
  --draft-tokens 3
```

---

## 7. API Usage

The server provides OpenAI- and Anthropic-compatible endpoints at `http://127.0.0.1:8080/v1`:

### OpenAI Chat Completions API
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Explain quantum entanglement in simple terms."}],
    "temperature": 0.7
  }'
```

### Anthropic Messages API
```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 8. Dual 3060 12GB & Concurrency FAQ

### Q1. Can `--max-concurrency 2` be used with `--max-context 131072`?
**Yes, absolutely.**
- NInfer uses a shared Paged KV Cache. When `--kv-capacity 131072` is allocated, 2 concurrent requests dynamically share this pool (e.g. 65K + 65K tokens, or 100K + 31K tokens).
- Concurrency 2 only adds ~350 MB of memory for extra GDN state slots and batch workspaces, fitting easily within ~20.7 GiB / 24.0 GiB.
- **TIP**: 만약 2개의 요청이 "동시에 각각 131K 토큰을 꽉 채워(총 262K)" 사용하는 극단적인 상황까지 VRAM을 최대한 확보하고 싶다면 `--kv-capacity auto`를 지정하면 남는 VRAM 전체(최대 약 170K~180K 토큰 풀)를 자동으로 KV 캐시에 할당합니다.

### Q2. Does Docker download the model automatically?
**No, you should download it on the host once.**
- The Docker image does not bake in the ~14.8 GiB weights to keep image size small.
- Run `bash scripts/download-qwen38.sh` on the host, and mount the folder into the container with `-v "$PWD/models:/workspace/models:ro"`.

