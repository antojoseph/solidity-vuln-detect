#!/bin/bash
# Full pipeline: RL rollouts + Agentic eval — all in parallel, all detached.
# Uses all 8 GPUs for vLLM data-parallel serving.
# Survives SSH disconnects.
#
# Usage:  bash ~/hud/run_all.sh
# Monitor: docker logs -f <container-name>
# Stop:    docker stop $(docker ps -q --filter name=solidity-)

set -e

echo "============================================================"
echo "Solidity Vuln Detection — Full Pipeline"
echo "============================================================"
echo ""
echo "Running in parallel:"
echo "  1. RL rollouts (3,834 scenarios x 6 samples = 23,004 rollouts)"
echo "  2. Agentic eval on SFT model (8,750 tasks with full repo access)"
echo ""
echo "Infrastructure:"
echo "  vLLM: 8x H100 data-parallel (Qwen3.5-9B-SFT)"
echo "  RL + Eval: CPU containers (scoring is deterministic)"
echo "============================================================"

MERGED_MODEL="/home/ubuntu/outputs/sft-9b-merged"
RL_OUTPUT="/home/ubuntu/outputs/rl"
AGENTIC_OUTPUT="/home/ubuntu/hud/results"
DATA_DIR="/home/ubuntu/hud/data"
HUD_DIR="/home/ubuntu/hud"

sudo mkdir -p "$RL_OUTPUT" "$AGENTIC_OUTPUT"
sudo chmod 777 "$RL_OUTPUT" "$AGENTIC_OUTPUT"

# Clean up any existing containers
echo "Cleaning up old containers..."
sg docker -c 'docker rm -f solidity-vllm solidity-rl solidity-agentic-sft solidity-standalone-sft 2>/dev/null' || true
sleep 2

# ============================================================
# Step 1: vLLM server — 8x data-parallel on all GPUs
# ============================================================
echo ""
echo "Step 1: Starting vLLM (8x data-parallel, all GPUs)..."
sg docker -c "docker run -d --name solidity-vllm \
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

echo "  Waiting for vLLM (8 replicas)..."
for i in $(seq 1 180); do
  if curl -s http://localhost:30002/v1/models > /dev/null 2>&1; then
    echo "  vLLM ready after $((i*5))s!"
    break
  fi
  if [ $i -eq 180 ]; then
    echo "  ERROR: vLLM failed to start"
    sg docker -c 'docker logs --tail 30 solidity-vllm'
    exit 1
  fi
  sleep 5
done

# ============================================================
# Step 2: RL rollouts (standalone eval format, all 3834 training scenarios)
# ============================================================
echo ""
echo "Step 2: Starting RL rollouts (3,834 scenarios x 6 samples)..."
sg docker -c "docker run -d --name solidity-rl \
  --network host \
  -v $HUD_DIR:/workspace/hud \
  -v $DATA_DIR:/workspace/hud/data:ro \
  -v $RL_OUTPUT:/workspace/outputs/rl \
  -e PYTHONPATH=/workspace/hud \
  -w /workspace/hud \
  solidity-train:sft \
  python3 /workspace/hud/run_online_rl.py \
    --model Qwen3.5-9B-SFT \
    --data /workspace/hud/data/skyrl_prompts.jsonl \
    --output /workspace/outputs/rl \
    --vllm-url http://localhost:30002/v1 \
    --max-scenarios 0 \
    --samples-per-scenario 6 \
    --batch-size 32 \
    --concurrency 64 \
    --max-steps 5 \
    --temperature 0.7"

# ============================================================
# Step 3: Agentic eval — SFT model on full repos (8,750 tasks)
# ============================================================
echo ""
echo "Step 3: Starting agentic eval (8,750 tasks with repo access)..."
sg docker -c "docker run -d --name solidity-agentic-sft \
  --network host \
  -v $DATA_DIR:/app/data:ro \
  -v $AGENTIC_OUTPUT:/app/results \
  hud-eval \
  --base-url http://localhost:30002/v1 \
  --model Qwen3.5-9B-SFT \
  --concurrency 16 \
  --max-steps 20"

# ============================================================
# Step 4: Standalone eval — full 972 tasks (for comparison consistency)
# ============================================================
echo ""
echo "Step 4: Starting standalone eval (972 tasks, SFT model)..."
sg docker -c "docker run -d --name solidity-standalone-sft \
  --network host \
  -v $HUD_DIR:/workspace/hud \
  -v $DATA_DIR:/workspace/hud/data:ro \
  -v $AGENTIC_OUTPUT:/workspace/hud/results \
  -e PYTHONPATH=/workspace/hud \
  -w /workspace/hud \
  solidity-train:sft \
  python3 /workspace/hud/run_eval_standalone.py \
    --base-url http://localhost:30002/v1 \
    --model Qwen3.5-9B-SFT \
    --concurrency 16 \
    --max-steps 5"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All jobs launched in detached Docker containers."
echo ""
echo "Containers:"
sg docker -c 'docker ps --format "  {{.Names}}\t{{.Status}}"'
echo ""
echo "Monitor:"
echo "  docker logs -f solidity-rl              # RL rollouts"
echo "  docker logs -f solidity-agentic-sft     # Agentic eval"
echo "  docker logs -f solidity-standalone-sft  # Standalone eval"
echo "  tail -f $RL_OUTPUT/progress.log         # RL progress"
echo "  watch nvidia-smi                        # GPU utilization"
echo ""
echo "Stop all:"
echo "  docker stop solidity-rl solidity-agentic-sft solidity-standalone-sft solidity-vllm"
echo "============================================================"
