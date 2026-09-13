# NInfer on Dual NVIDIA GeForce RTX 3060 12GB (Linux)

This guide covers running **Qwen3.8-27B** across **two NVIDIA GeForce RTX 3060 12GB GPUs** on Linux (Ubuntu / Bazzite / Fedora / Debian).

---

## Prerequisites

1. NVIDIA Driver >= 570
2. CUDA Toolkit >= 12.8
3. Two RTX 3060 12GB GPUs (SM86)

Check GPUs:
```bash
nvidia-smi
```

---

## Launchers

```bash
# 131K Context Mode
bash scripts/run-qwen38-3060x2-131k.sh

# 64K Single User Mode
bash scripts/run-qwen38-c1-3060x2.sh

# C8 Multi-user Mode
bash scripts/run-qwen38-c8-3060x2.sh
```

---

## CLI Inference

```bash
ninfer-cli models/qwen3_8_27b.ninfer \
  --prompt "Hello, tell me about yourself." \
  --devices 0,1 \
  --tensor-split 0.5,0.5 \
  --max-context 131072 \
  --kv-capacity 131072 \
  --kv-dtype int8 \
  --spec mtp \
  --draft-tokens 3
```
