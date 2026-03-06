"""Real Online RL training using SkyRL with weight updates.

Registers SolidityVulnEnv with SkyRL, builds the training config from
training.yaml, and launches the full RL loop:
  vLLM generates → SolidityVulnEnv scores → FSDP2 trainer updates weights → sync

This is the actual RL training (not just rollout collection).
Requires the GRPO Docker image with SkyRL + Ray + vLLM installed.

Usage:
    python3 run_skyrl_train.py \
        --model /workspace/outputs/sft-9b-merged \
        --data /workspace/hud/data/skyrl_prompts.jsonl \
        --output /workspace/outputs/rl-train \
        --config /workspace/hud/training.yaml
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_data_for_skyrl(input_path: str, output_path: str, env_id: str = "solidity-vuln"):
    """Convert skyrl_prompts.jsonl to SkyRL's expected format.

    SkyRL dataset needs each row to have extras.env_class matching the
    registered env ID, plus the prompt as a conversation.
    """
    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            extras = record.get("extras", {})
            extras["env_class"] = env_id

            # SkyRL expects prompt as conversation messages
            skyrl_record = {
                "prompt": record.get("prompt", []),
                "extras": extras,
            }
            fout.write(json.dumps(skyrl_record) + "\n")
            count += 1
    logger.info(f"Converted {count} prompts to SkyRL format at {output_path}")
    return count


def build_config(
    model_path: str,
    data_path: str,
    output_dir: str,
    config_path: str,
) -> dict:
    """Build the SkyRL config dict from our training.yaml.

    Uses TrajGym's config builder when available, falls back to manual
    construction from training.yaml.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    online_rl = config.get("online_rl", {})
    lora_cfg = config.get("lora", {})
    model_cfg = config.get("model", {})

    try:
        # Try using TrajGym's config builder (most complete)
        from trajgym.training.online_rl.config_builder import _build_skyrl_config
        logger.info("Using TrajGym config builder")
        skyrl_config = _build_skyrl_config(model_path, output_dir, config, data_path)
        return skyrl_config
    except ImportError:
        logger.info("TrajGym not available, building config manually")

    # Manual SkyRL config construction
    num_generations = int(online_rl.get("num_generations", 6))
    lora_rank = int(lora_cfg.get("r", 128))

    skyrl_config = {
        "model": {
            "model_path": model_path,
            "enable_lora": True,
            "lora_rank": lora_rank,
            "lora_alpha": int(lora_cfg.get("alpha", 256)),
            "lora_dropout": float(lora_cfg.get("dropout", 0)),
            "target_modules": lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            "trust_remote_code": True,
            "dtype": "bfloat16",
        },
        "generator": {
            "max_turns": int(online_rl.get("max_tool_calling_iterations", 10)),
            "max_prompt_length": int(online_rl.get("max_prompt_length", 16384)),
            "max_response_length": int(online_rl.get("max_completion_length", 8192)),
            "max_input_length": int(online_rl.get("max_prompt_length", 16384)),
            "num_generations": num_generations,
            "temperature": float(online_rl.get("generation_temperature", 0.7)),
            "top_p": float(online_rl.get("generation_top_p", 0.95)),
            "top_k": int(online_rl.get("generation_top_k", 20)),
            "stop": online_rl.get("generation_stop", ["</tool_call>", "<|im_end|>"]),
            "tool_call_format": online_rl.get("tool_call_format", "qwen3_coder"),
            "strip_think": bool(online_rl.get("strip_think", True)),
            "step_wise_trajectories": bool(online_rl.get("step_wise_trajectories", False)),
            "native_tool_schemas": bool(online_rl.get("native_tool_schemas", False)),
        },
        "trainer": {
            "learning_rate": float(online_rl.get("learning_rate", 5e-6)),
            "warmup_ratio": float(online_rl.get("warmup_ratio", 0.1)),
            "weight_decay": float(online_rl.get("weight_decay", 0.1)),
            "max_grad_norm": float(online_rl.get("max_grad_norm", 1.0)),
            "gradient_accumulation_steps": int(online_rl.get("gradient_accumulation_steps", 6)),
            "batch_size": int(online_rl.get("batch_size", 1)),
            "epochs": int(online_rl.get("epochs", 1)),
        },
        "algorithm": {
            "loss_type": online_rl.get("loss_type", "dapo"),
            "advantage_estimator": online_rl.get("advantage_estimator", "rloo"),
            "epsilon_high": float(online_rl.get("epsilon_high", 0.28)),
            "beta": float(online_rl.get("beta", 0.0)),
        },
        "data": {
            "data_path": data_path,
        },
        "output": {
            "output_dir": output_dir,
            "save_steps": int(config.get("output", {}).get("save_steps", 25)),
            "logging_steps": int(config.get("output", {}).get("logging_steps", 1)),
        },
    }
    return skyrl_config


def run_training(
    model_path: str,
    data_path: str,
    output_dir: str,
    config_path: str,
    env_id: str = "solidity-vuln",
):
    """Main training function — registers env and launches SkyRL."""

    # Convert data
    skyrl_data_path = os.path.join(output_dir, "skyrl_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    convert_data_for_skyrl(data_path, skyrl_data_path, env_id)

    # Build config
    config = build_config(model_path, skyrl_data_path, output_dir, config_path)
    logger.info(f"SkyRL config built. Model: {model_path}, Data: {skyrl_data_path}")

    # Load training.yaml for reward config
    with open(config_path) as f:
        training_config = yaml.safe_load(f)
    reward_config = training_config.get("reward", {})
    online_rl_cfg = training_config.get("online_rl", {})

    try:
        # Try TrajGym's full runtime (handles Ray, env registration, everything)
        from trajgym.training.online_rl.runtime import _run_skyrl_training
        logger.info("Using TrajGym SkyRL runtime")

        # Override the env class — TrajGym defaults to TrajGymTextEnv,
        # but we register SolidityVulnEnv instead.
        # We do this by monkey-patching the import in the Ray remote function.
        # The cleaner way is to pass agent_class, but SolidityVulnEnv
        # is a BaseTextEnv, not a StepAgent.

        # Actually, the cleanest approach: we register our env BEFORE
        # calling _run_skyrl_training, and modify the config to use our env_id.
        import ray
        from skyrl_gym.envs import register as skyrl_register
        from skyrl_env import SolidityVulnEnv

        # Pre-register our env (will be re-registered inside Ray remote)
        skyrl_register(
            id=env_id,
            entry_point=SolidityVulnEnv,
            kwargs={
                "max_turns": int(online_rl_cfg.get("max_tool_calling_iterations", 10)),
                "tool_call_format": online_rl_cfg.get("tool_call_format", "qwen3_coder"),
                "strip_think": bool(online_rl_cfg.get("strip_think", True)),
            },
        )
        logger.info(f"Registered {env_id} with SkyRL (entry_point=SolidityVulnEnv)")

        _run_skyrl_training(
            config=config,
            reward_config=reward_config,
            agent_class=None,  # Not using a StepAgent — our env IS the BaseTextEnv
            agent_kwargs=None,
            use_new_inference=bool(online_rl_cfg.get("use_new_inference", True)),
            trajectory_output_dir=os.path.join(output_dir, "trajectories"),
            pytorch_cuda_alloc_conf=online_rl_cfg.get("pytorch_cuda_alloc_conf"),
            horizon_schedule=online_rl_cfg.get("horizon_schedule"),
            hard_mask_statuses=online_rl_cfg.get("hard_mask_statuses", []),
            positive_only_until_step=int(online_rl_cfg.get("positive_only_until_step", 50)),
            positive_only_reward_floor=float(online_rl_cfg.get("positive_only_reward_floor", 0.0)),
        )

    except ImportError as e:
        logger.error(f"TrajGym runtime not available: {e}")
        logger.error("Install TrajGym or use the GRPO Docker image")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Real Online RL with SkyRL weight updates")
    parser.add_argument("--model", required=True, help="Path to SFT-merged model")
    parser.add_argument("--data", required=True, help="SkyRL prompts JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="training.yaml", help="Training config")
    args = parser.parse_args()

    logger.info(f"Starting Online RL training")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Data: {args.data}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Config: {args.config}")

    run_training(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
