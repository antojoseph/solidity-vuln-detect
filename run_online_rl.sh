#!/bin/bash
# Self-contained Online RL training script for Solidity vulnerability detection.
# Runs entirely in detached Docker containers — survives SSH disconnects.
#
# Uses ALL 8 GPUs: vLLM data-parallel (8 replicas of 9B model, ~17GB each)
# for maximum rollout throughput.
#
# Usage:
#   bash ~/hud/run_online_rl.sh
#
# Monitor:
#   docker logs -f solidity-rl-train
#   tail -f ~/outputs/rl/progress.log
#
# Stop:
#   docker stop solidity-rl-train solidity-rl-vllm

set -e

echo "============================================================"
echo "Solidity Vuln Detection — Online RL (RLOO)"
echo "============================================================"
echo "Model:    Qwen3.5-9B + SFT (merged)"
echo "Method:   RLOO (6 samples/prompt)"
echo "Data:     3834 training scenarios"
echo "Infra:    8x H100 — all GPUs for vLLM data-parallel inference"
echo "          RL script runs on CPU (scoring is deterministic)"
echo "============================================================"

MERGED_MODEL="/home/ubuntu/outputs/sft-9b-merged"
RL_OUTPUT="/home/ubuntu/outputs/rl"
DATA_DIR="/home/ubuntu/hud/data"
HUD_DIR="/home/ubuntu/hud"

# Create output dir
sudo mkdir -p "$RL_OUTPUT"
sudo chmod 777 "$RL_OUTPUT"

# Stop any existing containers
sg docker -c 'docker rm -f solidity-rl-vllm solidity-rl-train 2>/dev/null' || true

echo ""
echo "Step 1: Starting vLLM with 8x data-parallel replicas (all GPUs)..."
sg docker -c "docker run -d --name solidity-rl-vllm \
  --gpus all \
  --ipc=host \
  -p 30002:8000 \
  -v /home/ubuntu/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly \
  --model /workspace/outputs/sft-9b-merged \
  -dp 8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 32768 \
  --language-model-only \
  --enable-prefix-caching \
  --trust-remote-code \
  --served-model-name Qwen3.5-9B-SFT"

echo "  Waiting for vLLM to load model (8 replicas)..."
for i in $(seq 1 180); do
  if curl -s http://localhost:30002/v1/models > /dev/null 2>&1; then
    echo "  vLLM ready after $((i*5))s"
    break
  fi
  if [ $i -eq 180 ]; then
    echo "  ERROR: vLLM failed to start after 15 minutes"
    sg docker -c 'docker logs --tail 30 solidity-rl-vllm'
    exit 1
  fi
  sleep 5
done

echo ""
echo "Step 2: Starting RL rollout collection..."
echo "  Output: $RL_OUTPUT"
echo "  Logs:   docker logs -f solidity-rl-train"
echo ""

# RL script runs on CPU — all GPU memory is for vLLM inference.
# High concurrency (64) to saturate 8 data-parallel replicas.
sg docker -c "docker run -d --name solidity-rl-train \
  --network host \
  -v $HUD_DIR:/workspace/hud \
  -v $DATA_DIR:/workspace/hud/data:ro \
  -v $RL_OUTPUT:/workspace/outputs/rl \
  -v /home/ubuntu/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_BASE_URL=http://localhost:30002/v1 \
  -e PYTHONPATH=/workspace/hud \
  -w /workspace/hud \
  solidity-train:sft \
  python3 /workspace/hud/run_online_rl.py \
    --model Qwen3.5-9B-SFT \
    --data /workspace/hud/data/skyrl_prompts.jsonl \
    --output /workspace/outputs/rl \
    --config /workspace/hud/training.yaml \
    --vllm-url http://localhost:30002/v1 \
    --max-scenarios 0 \
    --samples-per-scenario 6 \
    --batch-size 32 \
    --concurrency 64"

echo ""
echo "============================================================"
echo "RL training started in detached Docker containers."
echo ""
echo "Monitor:"
echo "  docker logs -f solidity-rl-train"
echo "  tail -f $RL_OUTPUT/progress.log"
echo ""
echo "GPU utilization:"
echo "  watch nvidia-smi"
echo ""
echo "Stop:"
echo "  docker stop solidity-rl-train solidity-rl-vllm"
echo "============================================================"
