"""Test that the SkyRL Docker image has the deps needed for RL training."""
import sys

print("=== SkyRL Import Tests ===")

# SkyRL
from skyrl_gym.envs import register
from skyrl_gym.envs.base_text_env import BaseTextEnv
print("  skyrl_gym: OK")

# Our env
from skyrl_env import SolidityVulnEnv
assert issubclass(SolidityVulnEnv, BaseTextEnv)
print("  SolidityVulnEnv(BaseTextEnv): OK")

# SkyRL training
try:
    from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
    print("  skyrl_train (legacy): OK")
except ImportError:
    from skyrl.train.config import SkyRLGymConfig
    print(f"  skyrl.train (official, SkyRLGymConfig): OK")

# Ray
import ray
print(f"  ray: OK ({ray.__version__})")

# vLLM
import vllm
print(f"  vllm: OK ({vllm.__version__})")

# OmegaConf
from omegaconf import OmegaConf
print("  omegaconf: OK")

# TrajGym config builder
try:
    from trajgym.training.online_rl.config_builder import _build_skyrl_config
    print("  trajgym config_builder: OK")
except ImportError as e:
    print(f"  trajgym config_builder: MISSING ({e})")

# TrajGym runtime
try:
    from trajgym.training.online_rl.runtime import _run_skyrl_training
    print("  trajgym runtime: OK")
except ImportError as e:
    print(f"  trajgym runtime: MISSING ({e})")

# Our launcher
from run_skyrl_train import build_config, build_skyrl_overrides, convert_data_for_skyrl
print("  run_skyrl_train: OK")

# Test env registration
register(id="solidity-vuln-test", entry_point=SolidityVulnEnv, kwargs={"max_turns": 10})
print("  env registration: OK")

# Test episode
import json
with open("data/scenarios.json") as f:
    scenarios = json.load(f)
env = SolidityVulnEnv(extras={"scenario_id": scenarios[0]["id"], "max_turns": 5})
prompt, meta = env.init([])
r = env.step('<tool_call>\n{"name": "read_code", "arguments": {}}\n</tool_call>')
print(f"  episode test: done={r['done']}, reward={r['reward']}")

# Test config build
config = build_config(
    model_path="Qwen/Qwen3.5-9B",
    data_path="data/skyrl_prompts.jsonl",
    output_dir="/tmp/rl-test",
    config_path="training.yaml",
)
overrides = build_skyrl_overrides(config)
print(f"  config build: OK ({len(config)} resolved keys, {len(overrides)} overrides)")

print("\n=== All SkyRL tests passed. Ready for RL training. ===")
