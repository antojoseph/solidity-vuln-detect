#!/bin/bash
# Online RL with SkyRL — single colocated container.
# vLLM inference + FSDP trainer share the same process/GPU memory for
# log-probs and NCCL weight sync (required by SkyRL's RLOO algorithm).
#
# Architecture:
#   All 8 GPUs: SkyRL manages colocated vLLM engines + FSDP2 trainer
#   Image: solidity-train:qwen3-ready (torch 2.9.1, vllm 0.16.0, flash-attn rebuilt)
#   Model: Qwen/Qwen3-8B (base, no SFT — pure RL from base)
#
# Usage:  bash ~/hud/run_online_rl.sh
# Monitor: docker logs -f solidity-trainer
# Stop:    docker stop solidity-trainer

set -e

MODEL="Qwen/Qwen3-8B"
RL_OUTPUT="/home/ubuntu/outputs/rl-train-8b"
HUD_DIR="/home/ubuntu/hud"

sudo mkdir -p "$RL_OUTPUT" && sudo chmod 777 "$RL_OUTPUT"

# Clean up
sg docker -c 'docker rm -f solidity-trainer 2>/dev/null' || true

echo "============================================================"
echo "Starting SkyRL colocated training (all 8 GPUs)"
echo "  Model: $MODEL"
echo "============================================================"
sg docker -c "docker run -d --name solidity-trainer \
  --gpus all --ipc=host --shm-size=64g --network host \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH=/workspace/hud \
  -e WANDB_API_KEY=\"${WANDB_API_KEY}\" \
  -v $HUD_DIR:/workspace/hud \
  -v /home/ubuntu/outputs:/workspace/outputs \
  -v $HUD_DIR/data:/workspace/hud/data:ro \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace/hud \
  solidity-train:qwen3-ready \
  python3 run_skyrl_train.py \
    --model $MODEL \
    --data /workspace/hud/data/skyrl_prompts.jsonl \
    --output /workspace/outputs/rl-train-8b \
    --num-gpus 8"

echo ""
echo "============================================================"
echo "RL training launched!"
echo ""
echo "  Logs:  docker logs -f solidity-trainer"
echo "  GPUs:  watch nvidia-smi"
echo "  Stop:  docker stop solidity-trainer"
echo "============================================================"
