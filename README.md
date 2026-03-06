# Solidity Vulnerability Detection — RL Training Pipeline

Train open-source LLMs to find security vulnerabilities in Solidity smart contracts using SFT + Online RL. Built on real DeFi audit data from 10,600+ findings.

## Benchmark status

The checked-in result files include historical runs produced before the dataset and
SFT split leakage fixes in this repo. Treat those numbers as legacy diagnostics,
not benchmark-valid headline metrics.

To produce comparable results:

1. rebuild scenarios with `build_scenarios.py`
2. regenerate train-only SFT traces with `convert_traces_for_sft.py`
3. rerun evaluation on the regenerated held-out splits

## Integrity fixes in this repo

- `build_scenarios.py`
  - deduplicates exact `code_clean` matches
  - keeps each code / `finding_id` / `(protocol, source_file)` family on one side of the split
  - excludes any base scenario family that overlaps the held-out clean or OOD sets
- `convert_traces_for_sft.py`
  - defaults to `data/tasks_train.json`
  - filters out traces that are not in the allowed training split unless `--allow-all-scenarios` is used explicitly
- `run_online_rl.py`
  - is documented as rollout collection only
  - does not claim to do weight updates
- `run_skyrl_train.py`
  - is the intended launcher for actual online RL weight updates
- `run_eval_standalone.py` and `run_eval_agentic.py`
  - resume caches are keyed by provider + model + scenario
- `run_eval_agentic.py`
  - uses path ancestry checks instead of string-prefix path validation
  - separates strict tool-call scoring from assisted free-text extraction metrics
  - uses literal grep by default and restricts regex complexity
- `env.py`
  - parses line ranges such as `10-14`
  - does not inject a flat line score when the ground truth has no line annotations
  - prefers finding-level severity when available
- `skyrl_env.py`
  - no longer depends on provider SDK imports through `run_eval_standalone.py`

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

The scorer has strong support for the core canonical categories (reentrancy, oracle-manipulation, access-control, flash-loan, precision-rounding, slippage-protection, denial-of-service, integer-overflow, liquidation, and more), while the dataset may include additional canonical categories sourced from the raw audit corpus.

## Training Pipeline

```
Expert traces (Claude + Qwen3.5-397B)
       │
       ▼
  convert_traces_for_sft.py → structured tool_calls format
       │
       ▼
  ┌─────────────────────┐     ┌──────────────────────────┐
  │ Stage 1: SFT        │     │ Stage 2: Online RL       │
  │ run_sft.py          │────▶│ run_skyrl_train.py       │
  │ TRL + LoRA r=128    │     │ SkyRL / TrajGym training │
  │ train-only traces   │     │ weight updates + sync    │
  └─────────────────────┘     └──────────────────────────┘
       │                              │
       ▼                              ▼
  Eval: run_eval_standalone.py   W&B monitoring
  (held-out deterministic split) wandb_monitor.py
```

## Project Structure

```
├── audit_core.py                # Shared prompt/tool/episode logic (provider-independent)
├── env.py                       # Environment: 4 tools + 5-signal deterministic scorer
├── skyrl_env.py                 # SkyRL BaseTextEnv adapter (no provider SDK dependency)
├── run_eval_standalone.py       # Eval harness (Anthropic + OpenAI-compatible APIs)
├── run_eval_agentic.py          # Agentic eval (full repo exploration, 186 repos)
├── run_sft.py                   # SFT training (TRL SFTTrainer + LoRA)
├── run_online_rl.py             # Scored rollout collection / RLOO diagnostics
├── run_skyrl_train.py           # Actual online RL training with weight updates
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
│   ├── scenarios.json           # regenerated scenario corpus
│   ├── tasks_train.json         # grouped train split
│   ├── tasks_eval.json          # grouped held-out eval split
│   ├── tasks_eval_clean.json    # separate clean held-out split
│   ├── tasks_eval_ood.json      # separate OOD held-out split
│   ├── skyrl_prompts.jsonl      # SkyRL RL prompts
│   └── sft_traces_structured.jsonl  # train-only SFT data (structured tool_calls)
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

# Convert expert traces for SFT from train-only scenarios
python convert_traces_for_sft.py --task-file data/tasks_train.json --min-reward 0.7

# Optional sanity check
python test_smoke.py
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
# Real online RL training (weight updates)
python3 run_skyrl_train.py --model /path/to/model --data data/skyrl_prompts.jsonl --output outputs/rl-train

# Or collect scored rollout batches only
python3 run_online_rl.py --model /path/to/model --data data/skyrl_prompts.jsonl --output outputs/rl-rollouts

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
| `category_match` | 40% | Canonical category matching with keyword/group fallbacks |
| `explanation_quality` | 25% | Code references, attack terms, precondition overlap |
| `line_accuracy` | 15% | F1 score vs ground truth bug lines (±1 tolerance, skipped when GT lines are absent) |
| `exploitability` | 10% | Quality of attack_path, prerequisites, impact |
| `severity_match` | 10% | Distance from finding-level severity when present, else category default |

Penalties: 0.9× for missing structured fields, 0.85× for multi-label, 0.7× for using hints.

## Data Sources

- **[protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index)** — 10,600 DeFi audit findings across 30+ protocol types
- **[evmbench](https://github.com/openai/frontier-evals)** — 44 patched contracts (clean class) + 108 OOD vulnerable contracts from real audit contests

## Architecture Notes

- **HUD SDK optional** — `env.py` works standalone (lazy import). HUD integration available but not required.
- **Provider SDKs isolated from core env logic** — `audit_core.py` and `skyrl_env.py` can be imported without Anthropic installed.
- **Qwen3.5 native tool calling** — SFT traces use structured `tool_calls` fields. The tokenizer's `apply_chat_template` formats them as `qwen3_coder` XML natively.
- **vLLM nightly required** — Qwen3.5's hybrid DeltaNet/attention architecture needs vLLM from main branch (pre-0.17.0).
- **8x GPU data-parallel** — RL rollouts use `vllm serve -dp 8` for 8 replicas of the 9B model (~17GB each).

## Validation

Recommended local checks after changing the benchmark code:

```bash
python3 -m py_compile audit_core.py build_scenarios.py convert_traces_for_sft.py env.py \
  run_eval_standalone.py run_eval_agentic.py run_online_rl.py run_skyrl_train.py skyrl_env.py

python3 test_smoke.py
```

## License

Dataset: [protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index) (check repo for license)
Training framework: [Open Trajectory Gym](https://github.com/westonbrown/open-trajectory-gym) (Apache 2.0)
