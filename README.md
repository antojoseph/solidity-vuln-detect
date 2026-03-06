# Solidity Vulnerability Detection — RL Training Pipeline

Train open-source LLMs to find security vulnerabilities in Solidity smart contracts using SFT + Online RL. Built on real DeFi audit data from 10,600+ findings.

## Results

Qwen3.5-9B on 972 eval tasks:

| Model | Mean Reward | Median | Improvement |
|-------|-------------|--------|-------------|
| Qwen3.5-9B base | 0.437 | 0.385 | — |
| Qwen3.5-9B + SFT | **0.534** | **0.470** | **+22%** |
| Qwen3.5-397B (reference) | 0.503 | — | — |

SFT on 528 expert traces (Claude + Qwen3.5-397B) brings a 9B model to match or exceed a 44x larger model.

**Per-difficulty breakdown:**

| Difficulty | Base | SFT | Lift |
|-----------|------|-----|------|
| Easy (310) | 0.522 | 0.624 | +20% |
| Medium (316) | 0.444 | 0.532 | +20% |
| Hard (310) | 0.375 | 0.443 | +18% |

**Top SFT improvements by category:** oracle-manipulation (+0.18), stale-state (+0.16), locked-funds (+0.15), access-control (+0.14), slippage-protection (+0.12).

## How It Works

An LLM audits Solidity code using 4 tools, scored by a deterministic 5-signal reward function:

```
Agent                          Environment (env.py)
  │                                  │
  ├── read_code() ──────────────────▶│ Returns numbered Solidity snippet
  ├── get_context() ────────────────▶│ Protocol type + preconditions
  ├── list_hints() ─────────────────▶│ Detection heuristics (caps reward at 0.7)
  └── submit_finding(type, explain,  │
        severity, lines, ...) ──────▶│ Triggers deterministic scoring
                                     │
                                     ▼
                              Reward (0.0 - 1.0)
                              = 0.40 × category_match
                              + 0.25 × explanation_quality
                              + 0.15 × line_accuracy
                              + 0.10 × severity_match
                              + 0.10 × exploitability
```

22 vulnerability categories: reentrancy, oracle-manipulation, access-control, flash-loan, precision-rounding, slippage-protection, denial-of-service, integer-overflow, liquidation, and more.

## Training Pipeline

```
Expert traces (Claude + Qwen3.5-397B)
       │
       ▼
  convert_traces_for_sft.py → structured tool_calls format
       │
       ▼
  ┌─────────────────────┐     ┌──────────────────────┐
  │ Stage 1: SFT        │     │ Stage 2: Online RL   │
  │ run_sft.py           │────▶│ run_online_rl.py     │
  │ TRL + LoRA r=128     │     │ RLOO (6 samples/prompt)│
  │ 528 traces, 3 epochs │     │ 3,834 scenarios       │
  │ ~47 min on 1xH100    │     │ 8xH100 data-parallel  │
  └─────────────────────┘     └──────────────────────┘
       │                              │
       ▼                              ▼
  Eval: run_eval_standalone.py   W&B monitoring
  (972 tasks, deterministic)     wandb_monitor.py
```

## Project Structure

```
├── env.py                       # Environment: 4 tools + 5-signal deterministic scorer
├── skyrl_env.py                 # SkyRL BaseTextEnv adapter (bridges to Episode class)
├── run_eval_standalone.py       # Eval harness (Anthropic + OpenAI-compatible APIs)
├── run_eval_agentic.py          # Agentic eval (full repo exploration, 186 repos)
├── run_sft.py                   # SFT training (TRL SFTTrainer + LoRA)
├── run_online_rl.py             # Online RL rollouts (RLOO via vLLM API)
├── convert_traces_for_sft.py    # Convert Claude/Qwen traces → structured SFT format
├── prepare_skyrl_data.py        # Convert tasks → SkyRL prompt JSONL
├── training.yaml                # Training config (model, LoRA, SFT, RL params)
├── wandb_monitor.py             # Real-time W&B dashboard for RL monitoring
├── test_smoke.py                # 29-check validation suite
├── build_scenarios.py           # Data pipeline: findings → scenarios.json
├── Dockerfile.train             # Multi-stage NGC build (base/sft/grpo)
├── docker-compose.train.yml     # Docker services (smoke/sft/merge/grpo)
├── run_all.sh                   # Full pipeline launcher (survives SSH drops)
├── run_online_rl.sh             # RL-only launcher (detached containers)
├── data/                        # Generated datasets (gitignored)
│   ├── scenarios.json           # ~4,900 scenarios
│   ├── tasks_train.json         # 3,834 training tasks
│   ├── tasks_eval.json          # 972 eval tasks
│   ├── skyrl_prompts.jsonl      # SkyRL RL prompts
│   └── sft_traces_structured.jsonl  # SFT training data (structured tool_calls)
└── open-trajectory-gym/         # TrajGym framework (gitignored submodule)
```

## Quick Start

### Prerequisites

- Python 3.10+
- GPU with 80GB+ VRAM (H100 recommended) for training
- Docker + NVIDIA Container Toolkit for containerized runs
- vLLM nightly (`vllm/vllm-openai:nightly`) for Qwen3.5 inference

### 1. Generate Data

```bash
# Clone vulnerability data
git clone https://github.com/kadenzipfel/protocol-vulnerabilities-index.git /tmp/pvi

# Build scenarios
python build_scenarios.py /tmp/pvi

# Generate SkyRL prompts
python prepare_skyrl_data.py

# Convert expert traces for SFT (requires eval results in results/)
python convert_traces_for_sft.py --min-reward 0.7
```

### 2. Run Eval (Standalone)

```bash
# Against Anthropic API
python run_eval_standalone.py --model claude-sonnet-4-6 --max-tasks 50

# Against local vLLM server (Qwen3.5, OpenAI-compatible)
python run_eval_standalone.py --base-url http://localhost:30000/v1 \
  --model Qwen/Qwen3.5-9B --concurrency 8
```

### 3. Run SFT

```bash
# In Docker (recommended — handles all deps)
docker build -t solidity-train:sft --target sft -f Dockerfile.train .

docker run --rm --gpus device=0 --ipc=host --ulimit memlock=-1 \
  -v ./data:/workspace/hud/data:ro \
  -v ./outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  solidity-train:sft \
  python3 run_sft.py \
    --model Qwen/Qwen3.5-9B \
    --data /workspace/hud/data/sft_traces_structured.jsonl \
    --output /workspace/outputs/sft-9b
```

### 4. Run Online RL

```bash
# Full pipeline (vLLM + RL + agentic eval, all detached)
bash run_all.sh

# Monitor
docker logs -f solidity-rl
tail -f ~/outputs/rl/progress.log

# W&B dashboard
docker run -d --name solidity-wandb \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v ./outputs/rl:/workspace/rl:ro \
  solidity-train:sft \
  bash -c "pip install -q wandb && python3 wandb_monitor.py --rl-dir /workspace/rl"
```

### 5. Agentic Eval (Full Repo Exploration)

```bash
# Build agentic eval image
docker build -t hud-eval .

# Run against vLLM-served model
docker run -d --name agentic-eval --network host \
  -v ./data:/app/data:ro -v ./results:/app/results \
  hud-eval --base-url http://localhost:30000/v1 \
    --model Qwen3.5-9B-SFT --concurrency 16 --max-steps 20
```

## Scoring System

Deterministic, no LLM-as-judge. Five weighted components:

| Component | Weight | Method |
|-----------|--------|--------|
| `category_match` | 40% | Keyword matching against 22 canonical categories |
| `explanation_quality` | 25% | Code references, attack terms, precondition overlap |
| `line_accuracy` | 15% | F1 score vs ground truth bug lines (±1 tolerance) |
| `exploitability` | 10% | Quality of attack_path, prerequisites, impact |
| `severity_match` | 10% | Distance from expected severity level |

Penalties: 0.9× for missing structured fields, 0.85× for multi-label, 0.7× for using hints.

## Data Sources

- **[protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index)** — 10,600 DeFi audit findings across 30+ protocol types
- **[evmbench](https://github.com/openai/frontier-evals)** — 44 patched contracts (clean class) + 108 OOD vulnerable contracts from real audit contests

## Architecture Notes

- **HUD SDK optional** — `env.py` works standalone (lazy import). HUD integration available but not required.
- **Qwen3.5 native tool calling** — SFT traces use structured `tool_calls` fields. The tokenizer's `apply_chat_template` formats them as `qwen3_coder` XML natively.
- **vLLM nightly required** — Qwen3.5's hybrid DeltaNet/attention architecture needs vLLM from main branch (pre-0.17.0).
- **8x GPU data-parallel** — RL rollouts use `vllm serve -dp 8` for 8 replicas of the 9B model (~17GB each).

## License

Dataset: [protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index) (check repo for license)
Training framework: [Open Trajectory Gym](https://github.com/westonbrown/open-trajectory-gym) (Apache 2.0)
