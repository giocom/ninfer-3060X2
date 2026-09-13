# NInfer on Dual NVIDIA GeForce RTX 3060 12GB (Windows 11)

This guide covers running **Qwen3.8-27B** across **two NVIDIA GeForce RTX 3060 12GB GPUs** (total 24 GB VRAM) on Windows 11 with **tensor-split** and **131K+ token context window**.

---

## Hardware and VRAM Partitioning

- **GPU 0**: NVIDIA GeForce RTX 3060 12GB (SM86)
- **GPU 1**: NVIDIA GeForce RTX 3060 12GB (SM86)
- **Total Combined VRAM**: 24 GB
- **Tensor Split Ratio**: `--tensor-split 0.5,0.5` (or `--devices 0,1`)

### Memory Breakdown (Qwen3.8-27B)

| Component | GPU 0 (12GB) | GPU 1 (12GB) | Combined (24GB) |
|---|---:|---:|---:|
| Model Weights | ~7.4 GiB | ~7.4 GiB | ~14.8 GiB |
| KV Cache (131K INT8) | ~3.8 GiB | ~3.8 GiB | ~7.6 GiB |
| Workspace / CUDA Graphs | ~0.5 GiB | ~0.5 GiB | ~1.0 GiB |
| **Total VRAM Allocated** | **~11.7 GiB** | **~11.7 GiB** | **~23.4 GiB** |

---

## Quick Start Launchers

| Launcher | Description |
|---|---|
| `run-qwen38-3060x2-131k.bat` | **131K context window** (131,072 tokens), INT8 KV cache, MTP3, Dual-GPU |
| `run-qwen38-c1-3060x2.bat` | **64K context window**, lowest latency for single user |
| `run-qwen38-c8-3060x2.bat` | **8K context window**, up to 8 concurrent user requests |
| `benchmark-qwen38-3060x2.bat` | Benchmark throughput and latency on Dual 3060 |

---

## Manual Command Line

### 131K Context Mode (Dual 3060 12GB)

```bat
ninfer-serve.exe models\qwen3_8_27b.ninfer ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --devices 0,1 ^
  --tensor-split 0.5,0.5 ^
  --max-context 131072 ^
  --kv-capacity 131072 ^
  --max-concurrency 1 ^
  --max-pending-requests 16 ^
  --prefill-chunk 1024 ^
  --kv-dtype int8 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --lm-head-draft
```

### 64K Context Mode (Balanced)

```bat
ninfer-serve.exe models\qwen3_8_27b.ninfer ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --devices 0,1 ^
  --tensor-split 0.5,0.5 ^
  --max-context 65536 ^
  --kv-capacity 65536 ^
  --max-concurrency 1 ^
  --prefill-chunk 1024 ^
  --kv-dtype int8 ^
  --spec mtp ^
  --draft-tokens 3 ^
  --lm-head-draft
```

---

## API Compatibility

Once started, NInfer serves standard OpenAI- and Anthropic-compatible endpoints at:
- **Base URL**: `http://127.0.0.1:8080/v1`
- **Chat Completions**: `POST http://127.0.0.1:8080/v1/chat/completions`
- **Anthropic Messages**: `POST http://127.0.0.1:8080/v1/messages`
