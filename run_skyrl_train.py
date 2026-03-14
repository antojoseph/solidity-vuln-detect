#!/usr/bin/env python3
"""Launch SkyRL RLOO training for Solidity vulnerability detection.

Uses the official SkyRL API (Hydra CLI via main_base entrypoint).
Registers SolidityVulnEnv before training starts.

Usage (inside Docker):
    python3 run_skyrl_train.py \
        --model /workspace/outputs/sft-9b-merged \
        --data /workspace/hud/data/skyrl_prompts.jsonl \
        --output /workspace/outputs/rl-train \
        --num-gpus 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import runpy
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Keys that build_config() reads from online_rl.  Anything else at the
# top level of online_rl: (not inside algorithm/generator/trainer) is a
# mistake and will trigger a warning.
_KNOWN_ONLINE_RL_KEYS = {
    "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
    "warmup_ratio", "num_generations", "max_tool_calling_iterations",
    "max_completion_length", "max_prompt_length", "strip_think",
    # Sub-dicts that are passed through to SkyRL:
    "algorithm", "generator", "optimizer",
}


def convert_data_for_skyrl(input_path: str, output_dir: str, env_id: str = "solidity-vuln"):
    """Convert skyrl_prompts.jsonl to SkyRL parquet format."""
    import pandas as pd

    records = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            extras = record.get("extras", {})
            prompt = [{"role": "user", "content": "Begin your audit."}]
            records.append({
                "prompt": prompt,
                "env_class": env_id,
                "scenario_id": extras.get("scenario_id", ""),
                "difficulty": extras.get("difficulty", ""),
                "category": extras.get("category", ""),
            })

    df = pd.DataFrame(records)

    if "category" in df.columns and df["category"].nunique() > 1:
        df["_sort_key"] = df.groupby("category").cumcount()
        df = df.sample(frac=1, random_state=42)
        df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
        df = df.reset_index(drop=True)
        logger.info(
            "Category-stratified shuffle: %d categories interleaved",
            df["category"].nunique(),
        )

    out_path = os.path.join(output_dir, "train.parquet")
    df.to_parquet(out_path)
    logger.info(f"Converted {len(records)} prompts to {out_path}")
    return out_path, len(records)


def load_training_config(config_path: str) -> dict:
    if yaml is None:
        logger.warning("PyYAML not installed; using built-in defaults.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested dict into Hydra-style dot-separated key=value pairs.

    Example: {"a": {"b": 1, "c": {"d": 2}}} -> [("a.b", "1"), ("a.c.d", "2")]
    """
    items = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, key))
        elif isinstance(v, bool):
            items.append((key, "true" if v else "false"))
        elif isinstance(v, list):
            items.append((key, json.dumps(v)))
        else:
            items.append((key, str(v)))
    return items


def build_config(
    model_path: str,
    data_path: str,
    output_dir: str,
    config_path: str,
    num_gpus: int = 8,
    num_prompts: int = 3834,
    env_id: str = "solidity-vuln",
) -> dict:
    config = load_training_config(config_path)
    online_rl = config.get("online_rl", {})
    lora = config.get("lora", {})
    output = config.get("output", {})

    # Warn about unknown top-level keys in online_rl
    unknown = set(online_rl.keys()) - _KNOWN_ONLINE_RL_KEYS
    if unknown:
        logger.warning(
            "UNUSED online_rl keys (will be IGNORED): %s. "
            "Move algorithm params into online_rl.algorithm:, "
            "generator params into online_rl.generator:, "
            "or trainer params into online_rl.trainer:",
            sorted(unknown),
        )

    num_train_gpus = num_gpus

    return {
        "model_path": model_path,
        "data_path": data_path,
        "output_dir": output_dir,
        "env_id": env_id,
        "num_gpus": num_gpus,
        "num_train_gpus": num_train_gpus,
        "colocate_all": True,
        "epochs": int(online_rl.get("epochs", 1)),
        "batch_size": int(online_rl.get("batch_size", 1)),
        "gradient_accumulation_steps": int(online_rl.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(online_rl.get("learning_rate", 5e-6)),
        "warmup_ratio": float(online_rl.get("warmup_ratio", 0.1)),
        "num_prompts": num_prompts,
        "max_tool_calling_iterations": int(online_rl.get("max_tool_calling_iterations", 10)),
        "max_completion_length": int(online_rl.get("max_completion_length", 4096)),
        "max_prompt_length": int(online_rl.get("max_prompt_length", 8192)),
        "num_generations": int(online_rl.get("num_generations", 6)),
        "lora_rank": int(lora.get("r", 128)),
        "lora_alpha": int(lora.get("alpha", 256)),
        "save_steps": int(output.get("save_steps", 25)),
        "logging_steps": int(output.get("logging_steps", 1)),
        "strip_think": bool(online_rl.get("strip_think", True)),
        # Pass-through dicts — these use SkyRL's actual field names
        "algorithm": online_rl.get("algorithm", {}),
        "generator": online_rl.get("generator", {}),
        "optimizer": online_rl.get("optimizer", {}),
    }


def build_skyrl_overrides(resolved: dict) -> list[str]:
    overrides = [
        # Data
        f"data.train_data=['{resolved['data_path']}']",
        f"data.val_data=['{resolved['data_path']}']",
        # Model + LoRA
        f"trainer.policy.model.path={resolved['model_path']}",
        f"trainer.policy.model.lora.rank={resolved['lora_rank']}",
        f"trainer.policy.model.lora.alpha={resolved['lora_alpha']}",
        # Placement
        "trainer.strategy=fsdp2",
        f"trainer.placement.policy_num_gpus_per_node={resolved['num_train_gpus']}",
        f"trainer.placement.ref_num_gpus_per_node={resolved['num_train_gpus']}",
        f"trainer.placement.colocate_all={'true' if resolved['colocate_all'] else 'false'}",
        # Training loop
        f"trainer.epochs={resolved['epochs']}",
        f"trainer.train_batch_size={resolved['batch_size']}",
        f"trainer.policy_mini_batch_size={resolved['batch_size']}",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        f"trainer.update_epochs_per_batch={resolved['gradient_accumulation_steps']}",
        f"trainer.max_prompt_length={resolved['max_prompt_length']}",
        # Optimizer
        f"trainer.policy.optimizer_config.lr={resolved['learning_rate']}",
        f"trainer.policy.optimizer_config.num_warmup_steps={max(1, int(resolved['num_prompts'] / resolved['batch_size'] * resolved['warmup_ratio']))}",
        # Generator (fixed structural overrides)
        f"generator.inference_engine.num_engines={resolved['num_train_gpus']}",
        f"generator.max_turns={resolved['max_tool_calling_iterations']}",
        f"generator.n_samples_per_prompt={resolved['num_generations']}",
        f"generator.sampling_params.max_generate_length={resolved['max_completion_length']}",
        # Environment
        f"environment.env_class={resolved['env_id']}",
        # Checkpointing & logging
        f"trainer.ckpt_interval={resolved['save_steps']}",
        f"trainer.ckpt_path={resolved['output_dir']}/checkpoints",
        "trainer.eval_before_train=false",
        "trainer.resume_mode=none",
        "trainer.logger=wandb",
        "trainer.project_name=solidity-vuln-detect",
        "trainer.run_name=qwen3-8b-rl-v3",
    ]

    # Pass-through: algorithm.* -> trainer.algorithm.*
    for key, value in _flatten_dict(resolved.get("algorithm", {})):
        overrides.append(f"trainer.algorithm.{key}={value}")

    # Pass-through: generator.* -> generator.*
    for key, value in _flatten_dict(resolved.get("generator", {})):
        overrides.append(f"generator.{key}={value}")

    # Pass-through: optimizer.* -> trainer.policy.optimizer_config.*
    for key, value in _flatten_dict(resolved.get("optimizer", {})):
        overrides.append(f"trainer.policy.optimizer_config.{key}={value}")

    return overrides


def launch_training(resolved: dict) -> None:
    from skyrl_gym.envs.registration import registry
    from skyrl_gym.envs import register
    from skyrl_env import SolidityVulnEnv  # noqa: F401 — triggers auto-register

    env_id = resolved["env_id"]
    if env_id in {s.id for s in registry.values()}:
        del registry[env_id]
    register(
        id=env_id,
        entry_point=SolidityVulnEnv,
        kwargs={
            "max_turns": resolved["max_tool_calling_iterations"],
            "strip_think": resolved["strip_think"],
        },
    )
    logger.info("Registered %s env with kwargs: max_turns=%d", env_id, resolved["max_tool_calling_iterations"])

    overrides = build_skyrl_overrides(resolved)

    # Log all overrides so we can audit what actually reaches SkyRL
    logger.info("Launching SkyRL with %d overrides:", len(overrides))
    for ov in overrides:
        logger.info("  %s", ov)

    original_argv = sys.argv[:]
    try:
        sys.argv = ["skyrl.train.entrypoints.main_base", *overrides]
        runpy.run_module("skyrl.train.entrypoints.main_base", run_name="__main__")
    finally:
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description="SkyRL RLOO training")
    parser.add_argument("--model", required=True, help="Path to SFT-merged model")
    parser.add_argument("--data", required=True, help="SkyRL prompts JSONL")
    parser.add_argument("--output", required=True, help="Output/checkpoint directory")
    parser.add_argument("--config", default="training.yaml", help="Training config")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    data_path, num_prompts = convert_data_for_skyrl(args.data, args.output)
    resolved = build_config(
        model_path=args.model,
        data_path=data_path,
        output_dir=args.output,
        config_path=args.config,
        num_gpus=args.num_gpus,
        num_prompts=num_prompts,
    )
    launch_training(resolved)


if __name__ == "__main__":
    main()
