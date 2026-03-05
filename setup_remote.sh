#!/bin/bash
# Remote setup script for Lambda GPU machine
# Run this after rsync completes: ssh linux 'cd ~/hud && bash setup_remote.sh'

set -e

echo "=== Setting up Solidity Vuln Detection + TrajGym on $(hostname) ==="
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 1. Install system Python deps
echo -e "\n--- Installing Python dependencies ---"
pip install --upgrade pip
pip install skyrl-gym anthropic openai python-dotenv pyyaml jinja2

# 2. Install TrajGym from local source (for formatters, parsing, training CLI)
echo -e "\n--- Installing TrajGym (local) ---"
cd ~/hud/open-trajectory-gym
pip install -e ".[sft]" --no-deps 2>/dev/null || pip install -e . --no-deps 2>/dev/null || echo "TrajGym install skipped (may need manual setup)"
cd ~/hud

# 3. Verify imports
echo -e "\n--- Verifying imports ---"
python3 -c "
from skyrl_env import SolidityVulnEnv
from skyrl_gym.envs.base_text_env import BaseTextEnv
print(f'SolidityVulnEnv inherits BaseTextEnv: {issubclass(SolidityVulnEnv, BaseTextEnv)}')
"

# 4. Quick smoke test
echo -e "\n--- Running smoke test ---"
python3 -c "
import json
from skyrl_env import SolidityVulnEnv

with open('data/scenarios.json') as f:
    scenarios = json.load(f)
sid = scenarios[0]['id']

env = SolidityVulnEnv(extras={'scenario_id': sid})
prompt, meta = env.init([])
r = env.step('<tool_call>\n{\"name\": \"read_code\", \"arguments\": {}}\n</tool_call>')
r2 = env.step('<tool_call>\n{\"name\": \"submit_finding\", \"arguments\": {\"vulnerability_type\": \"oracle-manipulation\", \"explanation\": \"Stale price\", \"severity\": \"HIGH\", \"affected_lines\": \"5\"}}\n</tool_call>')
print(f'Smoke test: scenario={sid}, reward={r2[\"reward\"]:.4f}, done={r2[\"done\"]}')
"

# 5. Verify data
echo -e "\n--- Checking data ---"
wc -l data/skyrl_prompts.jsonl 2>/dev/null || echo "skyrl_prompts.jsonl not found — run: python3 prepare_skyrl_data.py"
wc -l data/slime_sft_traces.jsonl 2>/dev/null || echo "slime_sft_traces.jsonl not found"
ls -lh data/scenarios.json

echo -e "\n=== Setup complete ==="
echo "Next steps:"
echo "  1. python3 prepare_skyrl_data.py  (if skyrl_prompts.jsonl missing)"
echo "  2. SFT:  python3 -m trl sft --config training.yaml  (or use trajgym-train sft)"
echo "  3. RL:   see training.yaml for SkyRL config"
