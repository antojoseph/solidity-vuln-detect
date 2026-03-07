# Solidity Vulnerability Detection — Online RL Pipeline

Train LLMs to find smart contract vulnerabilities using online reinforcement learning. The model learns to audit Solidity code through multi-turn tool use, scored by a deterministic 5-signal reward function. Built on 10,600+ real DeFi audit findings.

## Results

| Model | Method | Standalone Eval | Agentic Eval |
|-------|--------|----------------|--------------|
| Qwen3.5-9B (base) | — | 0.437 | 0.416 |
| Qwen3.5-9B | SFT (LoRA r=128) | **0.534** (+22%) | 0.436 (+5%) |
| Qwen3-8B | Online RL (RLOO) | *training* | *training* |

Current RL run: avg_reward ~0.50-0.58, 8x H100, ~20h remaining.

## How It Works

An LLM agent audits Solidity code through multi-turn tool use. SkyRL's RLOO algorithm trains the model end-to-end: generate audit → score → compute advantage → update weights → sync to vLLM → repeat.

```
                    SkyRL Colocated Training Loop
                    ════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  vLLM Engines (8x)              FSDP2 Trainer              │
  │  ┌─────────────┐                ┌──────────────┐           │
  │  │ Generate     │  log-probs    │ RLOO policy  │           │
  │  │ audit        │──────────────▶│ gradient     │           │
  │  │ attempts     │               │ update       │           │
  │  │              │  NCCL weight  │              │           │
  │  │              │◀──────────────│              │           │
  │  └──────┬───────┘    sync       └──────────────┘           │
  │         │                                                   │
  │         ▼                                                   │
  │  SolidityVulnEnv (per episode)                             │
  │  ┌─────────────────────────────────────────┐               │
  │  │ Turn 1: read_code()    → Solidity snippet│               │
  │  │ Turn 2: get_context()  → protocol info   │               │
  │  │ Turn 3: <think>... reasoning             │               │
  │  │ Turn N: submit_finding(type, explain,    │               │
  │  │           severity, lines, attack_path)  │               │
  │  │                    │                     │               │
  │  │                    ▼                     │               │
  │  │         Deterministic Reward (0-1)       │               │
  │  └─────────────────────────────────────────┘               │
  │                                                             │
  │  All 8 GPUs — shared memory for log-probs + weight sync    │
  └─────────────────────────────────────────────────────────────┘
```

### Reward Function

Deterministic, no LLM-as-judge. Five weighted signals:

| Signal | Weight | Method |
|--------|--------|--------|
| `explanation_quality` | **30%** | Code references, attack terms, precondition overlap |
| `category_match` | 25% | Canonical vulnerability type matching with group fallbacks |
| `line_accuracy` | 20% | F1 score vs ground truth bug lines (±1 tolerance) |
| `exploitability` | 15% | Quality of attack_path, prerequisites, impact fields |
| `severity_match` | 10% | Distance from finding-level severity |

Penalties: 0.9× for missing structured fields, 0.85× for multi-label, 0.7× for using hints.
Reward shaping: 0.5× if model submits without calling `read_code()` first.

## Architecture

### Why Colocated (Not Separate Containers)

SkyRL's RLOO needs **log-probabilities** from vLLM (for the policy gradient) and **NCCL weight sync** (to push updated weights back to vLLM after each training step). Both require shared GPU memory between vLLM and the trainer — an external vLLM server can't provide this.

### Model Choice: Qwen3-8B

| Model | Architecture | vllm 0.16.0 | Notes |
|-------|-------------|-------------|-------|
| Qwen3-8B | `Qwen3ForCausalLM` | ✅ Native | Text-only, instruct model with tool calling |
| Qwen3.5-9B | `Qwen3_5ForConditionalGeneration` | ❌ | VLM (multimodal), needs vllm 0.17+ |

Qwen3-8B IS the instruct model (not `Qwen3-8B-Instruct` — that doesn't exist). The base model is `Qwen3-8B-Base`.

### Critical: Multi-Turn Mode

```yaml
generator.batched=false   # REQUIRED — enables multi-turn agent_loop
generator.batched=true    # BROKEN — single-turn, model can't use tools, all rewards = 0
```

With `batched=true`, the model generates one response and the episode ends. It can't call `read_code()` then `submit_finding()` in separate turns. All rewards are zero and no learning occurs.

## Training Configuration

```yaml
# Current best config (training.yaml)
model: Qwen/Qwen3-8B
method: RLOO (online RL, no SFT prerequisite)
hardware: 8x H100 80GB

online_rl:
  batch_size: 4              # 4 scenarios per step (diversity)
  num_generations: 8          # 8 rollouts per scenario (RLOO variance reduction)
  # Total: 32 rollouts/step across 4 different vulnerabilities
  learning_rate: 5.0e-6
  warmup_steps: 383           # 10% of total steps
  max_tool_calling_iterations: 10
  max_completion_length: 8192
  advantage_estimator: rloo
  beta: 0.0                   # No KL penalty (maximize exploration)

lora:
  r: 128
  alpha: 256
  target_modules: all-linear
```

## Reproducing From Scratch

This section covers everything needed to recreate the training infrastructure on a fresh machine.

### Hardware Requirements

- **GPUs**: 8x NVIDIA H100 80GB (or A100 80GB). All 8 are used simultaneously.
  - ~71GB VRAM per GPU during training (model + KV cache + optimizer states)
  - Fewer GPUs: adjust `--num-gpus`, `num_engines`, and batch math accordingly
- **RAM**: 128GB+ system memory (Ray overhead + data loading)
- **Disk**: 100GB+ for Docker images (~35GB) + model weights (~16GB) + checkpoints
- **Network**: Internet access for HuggingFace model download (~16GB for Qwen3-8B)

### Step 1: Server Setup

```bash
# Tested on Ubuntu 22.04 with NVIDIA driver 570+
# Install Docker + NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi

# Clone this repo
git clone <repo-url> ~/hud
cd ~/hud
```

### Step 2: Build the Docker Image (from scratch)

The image is built in layers. The base is Anyscale's Ray image with CUDA 12.8.

```bash
# --- Layer 1: Base image with Ray + CUDA ---
docker pull novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8

# --- Layer 2: Install CUDA toolkit + system deps ---
docker run -d --name skyrl-build --gpus all \
  novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 sleep infinity

docker exec skyrl-build bash -c '
  sudo apt-get update -y && sudo apt-get install -y wget build-essential libnuma-dev
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
  sudo sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit
  rm -f cuda_12.8.0_570.86.10_linux.run
'

# --- Layer 3: Install PyTorch + flash-attn + core ML deps ---
docker exec skyrl-build bash -c '
  pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

  # flash-attn MUST be built from source against the installed torch
  pip install flash-attn --no-build-isolation --no-deps \
    --force-reinstall --no-cache-dir --no-binary flash-attn

  # ML dependencies (pinned for compatibility)
  pip install --no-deps \
    anthropic openai peft accelerate transformers==4.57.6 \
    wandb jaxtyping beartype torchdata loguru safetensors \
    huggingface_hub tokenizers regex

  pip install \
    python-dotenv pyyaml jinja2 pandas pyarrow \
    sentry-sdk httpx httpcore jiter docstring-parser \
    annotated-doc typer shellingham hf-xet wadler-lindig \
    omegaconf hydra-core datasets scipy==1.14.1
'

# --- Layer 4: Install SkyRL + skyrl-gym ---
docker exec skyrl-build bash -c '
  git clone --depth 1 https://github.com/novasky-ai/SkyRL.git /home/ray/SkyRL
  pip install --no-deps -e /home/ray/SkyRL
  pip install --force-reinstall --no-deps /home/ray/SkyRL/skyrl-gym

  # Install vllm (must match torch version — 0.16.0 for torch 2.9.1)
  pip install vllm==0.16.0 --extra-index-url https://wheels.vllm.ai/nightly
'

# --- Layer 5: Verify everything works ---
docker exec skyrl-build python3 -c '
import torch; print(f"torch: {torch.__version__}")
from flash_attn.bert_padding import pad_input; print("flash_attn: OK")
import vllm; print(f"vllm: {vllm.__version__}")
from skyrl.train.entrypoints.main_base import BasePPOExp; print("skyrl: OK")
from skyrl_gym.envs import register; print("skyrl_gym: OK")
print("ALL PASS")
'

# --- Layer 6: Patch skyrl_gym for custom env auto-registration ---
docker exec skyrl-build bash -c 'cat >> /home/ray/anaconda3/lib/python3.12/site-packages/skyrl_gym/__init__.py << "PATCH"

# Auto-import user envs from /workspace/hud
import sys as _sys
if "/workspace/hud" not in _sys.path:
    _sys.path.insert(0, "/workspace/hud")
try:
    import skyrl_env  # registers solidity-vuln
except ImportError:
    pass
PATCH
'

# --- Commit final image ---
docker commit skyrl-build solidity-train:qwen3-ready
docker rm -f skyrl-build
```

**Timing**: ~15 min total (flash-attn compilation is ~1 min on H100, the rest is downloads).

### Step 3: Download the Model

```bash
# Qwen3-8B is ~16GB. Ensure HF cache permissions are correct.
sudo chmod -R 777 ~/.cache/huggingface/

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B')
print('Done')
"
```

> **Note**: `Qwen/Qwen3-8B` IS the instruct model. The base model is `Qwen/Qwen3-8B-Base`. There is no `Qwen3-8B-Instruct`.

### Step 4: Prepare Data

```bash
cd ~/hud

# Option A: Build from raw audit data
git clone https://github.com/kadenzipfel/protocol-vulnerabilities-index.git /tmp/pvi
python build_scenarios.py /tmp/pvi
python prepare_skyrl_data.py

# Option B: Use pre-built data (if data/ directory already exists)
ls data/scenarios.json data/skyrl_prompts.jsonl
# Should show: 4,914 scenarios, 3,834 RL prompts
```

### Step 5: Configure and Launch

```bash
# Set W&B key
export WANDB_API_KEY=your_wandb_key
echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc

# Edit training.yaml if needed (defaults are tuned for 8x H100)
# Key params: batch_size, num_generations, learning_rate, max_tool_calling_iterations

# Launch training
bash run_online_rl.sh
```

### Step 6: Monitor

```bash
# Live logs
docker logs -f solidity-trainer

# GPU utilization (should be ~100% on all 8, ~71GB/81GB memory)
watch nvidia-smi

# W&B dashboard (metrics appear after first completed step, ~2 min)
# Look for: reward/avg_raw_reward trending up, policy/grad_norm > 0

# Quick health check
docker logs solidity-trainer 2>&1 | grep "avg_final_rewards" | tail -5
# Healthy: values > 0 and gradually increasing
# Dead run: all 0.0 (check generator.batched setting)
```

### Step 7: Evaluate After Training

```bash
# Serve the trained model with vLLM
docker run -d --name vllm-eval --gpus all -p 8000:8000 \
  -v ~/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:nightly \
  --model /workspace/outputs/rl-train-8b/checkpoints/latest \
  --max-model-len 32768 --trust-remote-code

# Run standalone eval
python run_eval_standalone.py \
  --base-url http://localhost:8000/v1 \
  --model rl-checkpoint --concurrency 8

# Run agentic eval (full repo exploration)
python run_eval_agentic.py \
  --base-url http://localhost:8000/v1 \
  --model rl-checkpoint --concurrency 16 --max-steps 20
```

## Troubleshooting

### Container Build Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `flash_attn_2_cuda...undefined symbol` | flash-attn compiled for wrong torch | Rebuild: `pip install flash-attn --no-build-isolation --no-deps --no-cache-dir --no-binary flash-attn` |
| `pip install flash-attn` upgrades torch | pip resolves deps and pulls torch 2.10 | Always use `--no-deps` |
| `Using cached flash_attn...whl` | Stale wheel in pip cache | Add `--no-cache-dir --no-binary flash-attn` |
| vllm import error after upgrade | vllm binary linked to wrong torch | Don't upgrade vllm. Use 0.16.0 with torch 2.9.1 |
| `No module named 'skyrl'` | SkyRL not installed as editable | `pip install --no-deps -e /home/ray/SkyRL` |

### Training Launch Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `WANDB_API_KEY is required` | Key not passed to container | `export WANDB_API_KEY=... && bash run_online_rl.sh` |
| `Invalid policy_mini_batch_size_per_gpu: 0` | batch × samples not divisible by GPUs | Ensure `batch_size * num_generations >= num_gpus` |
| `num_policy_gpus != num_rollout_gpus` | num_engines != num_gpus with colocate_all | Set `num_engines = num_gpus` |
| `Invalid fields {'warmup_ratio'}` | SkyRL uses `num_warmup_steps` (int) | Compute steps manually: `int(total_steps * ratio)` |
| `Already registered (id=solidity-vuln)` | Double registration | Guard with registry check before register() |
| `No registered env with id: solidity-vuln` | Env not registered in Ray worker | Apply skyrl_gym init patch (Step 2, Layer 6) |
| `Model architectures not supported` | vllm doesn't know the model | Use Qwen3-8B (supported), not Qwen3.5-9B |

### Training Runtime Issues

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| `avg_final_rewards: 0.0` forever | `generator.batched=true` | Set `generator.batched=false` in run_skyrl_train.py |
| `grad_norm: 0.0` forever | No reward signal | Check batched setting + verify env.step() is called |
| Rewards 0 but env.step() logging shows | Submissions all wrong | Check if model reads code before submitting |
| OOM on GPU | batch_size too large or max_model_len too high | Reduce batch_size or gpu_memory_utilization |
| Training very slow (>5 min/step) | Long multi-turn episodes | Reduce max_tool_calling_iterations or max_completion_length |
| `EngineCore initialization failed` | vllm can't load model | Check model architecture support + GPU memory |

### Verifying the Container

Run this smoke test before launching training:

```bash
docker run --rm --gpus all --ipc=host \
  -v ~/hud:/workspace/hud \
  -v ~/outputs:/workspace/outputs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/workspace/hud \
  solidity-train:qwen3-ready \
  python3 -c "
import torch; print('torch:', torch.__version__)
from flash_attn.bert_padding import pad_input; print('flash_attn: OK')
import vllm; print('vllm:', vllm.__version__)
import skyrl; print('skyrl: OK')
import skyrl_gym; skyrl_gym.pprint_registry()  # should show solidity-vuln
from skyrl_env import SolidityVulnEnv; print('env: OK')
import json; s = json.load(open('/workspace/hud/data/scenarios.json'))
print(f'scenarios: {len(s)}')
print('ALL CHECKS PASSED')
"
```

Expected output:
```
torch: 2.9.1+cu128
flash_attn: OK
vllm: 0.16.0
skyrl: OK
aime  gsm8k  gsm8k_multi_turn  lcb  search  searchcode  solidity-vuln  text2sql
env: OK
scenarios: 4914
ALL CHECKS PASSED
```

## Project Structure

```
├── run_online_rl.sh             # Launch script (single colocated container, 8 GPUs)
├── run_skyrl_train.py           # SkyRL config builder + Hydra override generator
├── skyrl_env.py                 # SolidityVulnEnv (BaseTextEnv adapter)
├── training.yaml                # Hyperparameters (SFT + RL sections)
│
├── env.py                       # Environment: 4 tools + 5-signal scorer
├── run_eval_standalone.py       # Eval harness + Episode class + reward weights
├── run_eval_agentic.py          # Agentic eval (full repo exploration, 186 repos)
│
├── run_sft.py                   # SFT training (TRL SFTTrainer + LoRA)
├── convert_traces_for_sft.py    # Expert traces → structured SFT format
├── prepare_skyrl_data.py        # Tasks → SkyRL prompt JSONL
├── build_scenarios.py           # Raw findings → deduplicated scenarios.json
│
├── wandb_monitor.py             # Real-time W&B dashboard
├── test_smoke.py                # Validation suite
│
├── data/
│   ├── scenarios.json           # 4,914 vulnerability scenarios
│   ├── skyrl_prompts.jsonl      # 3,834 RL training prompts
│   ├── tasks_train.json         # Train split
│   ├── tasks_eval.json          # Held-out eval split
│   └── sft_traces_structured.jsonl
│
└── open-trajectory-gym/         # TrajGym framework (submodule)
```

## Dependency Stack

The exact versions matter — ABI mismatches between torch/vllm/flash-attn cause silent crashes.

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.9.1+cu128 | Pinned by vllm 0.16.0 |
| vllm | 0.16.0 | Native Qwen3ForCausalLM support |
| flash-attn | 2.8.3 | **Must rebuild from source** against torch 2.9.1 |
| transformers | 4.57.6 | Supports `qwen3` model_type |
| skyrl | 0.3.0 | At `/home/ray/SkyRL/` in container |
| skyrl_gym | 0.0.0 | Patched for custom env auto-import |
| Python | 3.12 | Ray + CUDA 12.8.1 |

### flash-attn: The Most Common Failure

Pre-built wheels are compiled for a specific torch version. Installing flash-attn without `--no-deps --no-binary` will either:
1. Pull a wheel compiled for torch 2.10 → ABI crash at import time
2. Upgrade torch to 2.10 → breaks vllm 0.16.0 (needs torch==2.9.1)

```bash
# The correct command (rebuild from source, don't touch torch):
pip install flash-attn --no-build-isolation --no-deps --force-reinstall --no-cache-dir --no-binary flash-attn
```

The error when it's wrong: `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`

## Env Registration

SkyRL runs training in a Ray remote function (`skyrl_entrypoint`) — a separate process. Custom env registration in the launcher doesn't propagate to Ray workers.

**Solution:** Patch `skyrl_gym/__init__.py` in the container to auto-import our env:

```python
# Append to /home/ray/anaconda3/lib/python3.12/site-packages/skyrl_gym/__init__.py
import sys as _sys
if "/workspace/hud" not in _sys.path:
    _sys.path.insert(0, "/workspace/hud")
try:
    import skyrl_env  # auto-registers solidity-vuln
except ImportError:
    pass
```

`skyrl_env.py` calls `skyrl_gym.register(id="solidity-vuln", ...)` at module level, so importing it in any process registers the env.

## Key Learnings

### What Makes RL Training Work

1. **`generator.batched=false`** — enables multi-turn tool use (the most critical setting)
2. **Instruct model** — base models can't produce valid tool calls, resulting in all-zero rewards
3. **batch_size ≥ 4** — diversity across scenarios per gradient step (batch_size=1 is too narrow)
4. **Warmup** — SkyRL uses `num_warmup_steps` (integer), not `warmup_ratio`
5. **Reward shaping** — penalize blind submissions (no `read_code()` → reward × 0.5)

### What Causes Dead Runs (All Rewards = 0)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `avg_final_rewards: 0.0`, `grad_norm: 0.0` | `batched=true` (single-turn) | Set `generator.batched=false` |
| Same as above | Base model can't tool-call | Use instruct model or SFT first |
| `zero_variance_filter` discarding everything | All samples score identically | Increase batch_size for diversity |
| `env.step()` never called | batched mode skips multi-turn loop | Set `generator.batched=false` |

### Batch Size Math

```
policy_mini_batch_size × n_samples_per_prompt ≥ dp_size (num GPUs)

Example: batch=4, samples=8, GPUs=8 → 4×8/8 = 4 per GPU ✓
Example: batch=1, samples=6, GPUs=8 → 1×6/8 = 0.75 → FAILS
```

Also: `colocate_all=true` requires `num_engines == num_policy_gpus`.

### Healthy Training Indicators

| Metric | Healthy Range | Concern |
|--------|--------------|---------|
| avg_final_rewards | > 0, trending up | 0.0 = dead run |
| policy_entropy | 0.05 - 0.12 | < 0.01 = collapsed, > 0.5 = random |
| grad_norm | 0.001 - 0.05 | 0.0 = no learning, > 1.0 = unstable |
| clip_ratio | 0.005 - 0.05 | 0.0 = no updates, > 0.2 = too aggressive |
| response_length | increasing early on | model learning to use more turns |

## W&B Metrics

SkyRL logs automatically:

| Panel | Metrics |
|-------|---------|
| Rewards | `reward/avg_raw_reward`, `reward/avg_pass_at_8`, `reward/mean_positive_reward` |
| Loss | `loss/avg_final_rewards`, `loss/avg_raw_advantages` |
| Policy | `policy/final_loss`, `policy/policy_entropy`, `policy/clip_ratio`, `policy/grad_norm`, `policy/policy_lr` |
| Generation | `generate/avg_num_tokens`, `generate/min_num_tokens`, `generate/max_num_tokens` |
| Timing | `timing/generate`, `timing/policy_train`, `timing/sync_weights` |
| Environment | `environment/solidity-vuln/subscore/*`, `environment/solidity-vuln/turns`, `environment/solidity-vuln/read_code_called` |

## Data Integrity

- `build_scenarios.py` deduplicates exact code matches and keeps each finding family on one side of the train/eval split
- `convert_traces_for_sft.py` filters traces to training split only (unless `--allow-all-scenarios`)
- Eval caches are keyed by provider + model + scenario to prevent cross-contamination

## Data Sources

- **[protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index)** — 10,600 DeFi audit findings across 30+ protocol types
- **[evmbench](https://github.com/openai/frontier-evals)** — 44 patched contracts (clean class) + 108 OOD contracts

## License

Dataset: [protocol-vulnerabilities-index](https://github.com/kadenzipfel/protocol-vulnerabilities-index) (check repo for license)
Training framework: [SkyRL](https://github.com/NovaSky-AI/SkyRL) / [Open Trajectory Gym](https://github.com/westonbrown/open-trajectory-gym) (Apache 2.0)
