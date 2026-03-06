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


def convert_data_for_skyrl(input_path: str, output_dir: str, env_id: str = "solidity-vuln"):
    """Convert skyrl_prompts.jsonl to SkyRL parquet format.

    SkyRL expects:
    - prompt: list of message dicts [{"role": "user", "content": "..."}]
    - env_class: string at top level (not nested)
    - scenario metadata flattened as top-level columns
    """
    import pandas as pd

    records = []
    with open(input_path) as f:
        for line in f:
            record = json.loads(line)
            extras = record.get("extras", {})
            # SkyRL prompt format: just user message (env adds system prompt in init())
            prompt = [{"role": "user", "content": "Begin your audit."}]
            records.append({
                "prompt": prompt,
                "env_class": env_id,
                "scenario_id": extras.get("scenario_id", ""),
                "difficulty": extras.get("difficulty", ""),
                "category": extras.get("category", ""),
            })

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "train.parquet")
    df.to_parquet(out_path)
    logger.info(f"Converted {len(records)} prompts to {out_path}")
    return out_path


def load_training_config(config_path: str) -> dict:
    if yaml is None:
        logger.warning("PyYAML not installed; using built-in SkyRL launcher defaults.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def build_config(
    model_path: str,
    data_path: str,
    output_dir: str,
    config_path: str,
    num_gpus: int = 8,
    env_id: str = "solidity-vuln",
) -> dict:
    config = load_training_config(config_path)
    online_rl = config.get("online_rl", {})
    lora = config.get("lora", {})
    output = config.get("output", {})

    if num_gpus < 2:
        num_inference_gpus = 1
        num_train_gpus = 1
        colocate_all = True
    else:
        num_inference_gpus = max(1, num_gpus // 2)
        num_train_gpus = max(1, num_gpus - num_inference_gpus)
        colocate_all = False

    return {
        "model_path": model_path,
        "data_path": data_path,
        "output_dir": output_dir,
        "env_id": env_id,
        "num_gpus": num_gpus,
        "num_inference_gpus": num_inference_gpus,
        "num_train_gpus": num_train_gpus,
        "colocate_all": colocate_all,
        "epochs": int(online_rl.get("epochs", 1)),
        "batch_size": int(online_rl.get("batch_size", 1)),
        "gradient_accumulation_steps": int(online_rl.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(online_rl.get("learning_rate", 5e-6)),
        "max_tool_calling_iterations": int(online_rl.get("max_tool_calling_iterations", 10)),
        "max_completion_length": int(online_rl.get("max_completion_length", 4096)),
        "max_prompt_length": int(online_rl.get("max_prompt_length", 8192)),
        "num_generations": int(online_rl.get("num_generations", 6)),
        "generation_temperature": float(online_rl.get("generation_temperature", 0.7)),
        "generation_top_p": float(online_rl.get("generation_top_p", 0.95)),
        "generation_top_k": int(online_rl.get("generation_top_k", 20)),
        "generation_stop": online_rl.get("generation_stop", ["</tool_call>", "<|im_end|>"]),
        "advantage_estimator": online_rl.get("advantage_estimator", "rloo"),
        "beta": float(online_rl.get("beta", 0.0)),
        "lora_rank": int(lora.get("r", 128)),
        "lora_alpha": int(lora.get("alpha", 256)),
        "save_steps": int(output.get("save_steps", 25)),
        "logging_steps": int(output.get("logging_steps", 1)),
        "strip_think": bool(online_rl.get("strip_think", True)),
    }


def build_skyrl_overrides(resolved: dict) -> list[str]:
    stop_strings = json.dumps(resolved["generation_stop"])
    return [
        f"data.train_data=['{resolved['data_path']}']",
        f"data.val_data=['{resolved['data_path']}']",
        f"trainer.policy.model.path={resolved['model_path']}",
        f"trainer.policy.model.lora.rank={resolved['lora_rank']}",
        f"trainer.policy.model.lora.alpha={resolved['lora_alpha']}",
        "trainer.strategy=fsdp2",
        f"trainer.placement.policy_num_gpus_per_node={resolved['num_train_gpus']}",
        f"trainer.placement.ref_num_gpus_per_node={resolved['num_train_gpus']}",
        f"trainer.placement.colocate_all={'true' if resolved['colocate_all'] else 'false'}",
        f"trainer.epochs={resolved['epochs']}",
        f"trainer.train_batch_size={resolved['batch_size']}",
        f"trainer.policy_mini_batch_size={resolved['batch_size']}",
        "trainer.micro_forward_batch_size_per_gpu=1",
        "trainer.micro_train_batch_size_per_gpu=1",
        f"trainer.update_epochs_per_batch={resolved['gradient_accumulation_steps']}",
        f"trainer.max_prompt_length={resolved['max_prompt_length']}",
        f"trainer.policy.optimizer_config.lr={resolved['learning_rate']}",
        f"trainer.algorithm.advantage_estimator={resolved['advantage_estimator']}",
        f"trainer.algorithm.use_kl_loss={'true' if resolved['beta'] > 0 else 'false'}",
        f"trainer.algorithm.use_kl_in_reward={'true' if resolved['beta'] > 0 else 'false'}",
        f"generator.inference_engine.num_engines={resolved['num_inference_gpus']}",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.async_engine=true",
        "generator.inference_engine.gpu_memory_utilization=0.85",
        "generator.batched=true",
        f"generator.max_turns={resolved['max_tool_calling_iterations']}",
        f"generator.n_samples_per_prompt={resolved['num_generations']}",
        f"generator.sampling_params.max_generate_length={resolved['max_completion_length']}",
        f"generator.sampling_params.temperature={resolved['generation_temperature']}",
        f"generator.sampling_params.top_p={resolved['generation_top_p']}",
        f"generator.sampling_params.top_k={resolved['generation_top_k']}",
        f"generator.sampling_params.stop={stop_strings}",
        f"environment.env_class={resolved['env_id']}",
        f"trainer.ckpt_interval={resolved['save_steps']}",
        f"trainer.ckpt_path={resolved['output_dir']}/checkpoints",
        "trainer.eval_before_train=false",
        "trainer.logger=wandb",
        "trainer.project_name=solidity-vuln-detect",
        f"trainer.default_hdfs_dir={resolved['output_dir']}",
    ]


def launch_training(resolved: dict) -> None:
    from skyrl_gym.envs import register
    from skyrl_env import SolidityVulnEnv

    register(
        id=resolved["env_id"],
        entry_point=SolidityVulnEnv,
        kwargs={
            "max_turns": resolved["max_tool_calling_iterations"],
            "strip_think": resolved["strip_think"],
        },
    )
    logger.info("Registered %s env with SkyRL", resolved["env_id"])

    overrides = build_skyrl_overrides(resolved)
    argv = ["skyrl.train.entrypoints.main_base", *overrides]
    logger.info("Launching SkyRL with %d overrides", len(overrides))

    original_argv = sys.argv[:]
    try:
        sys.argv = argv
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

    data_path = convert_data_for_skyrl(args.data, args.output)
    resolved = build_config(
        model_path=args.model,
        data_path=data_path,
        output_dir=args.output,
        config_path=args.config,
        num_gpus=args.num_gpus,
    )
    logger.info(
        "Launching SkyRL RLOO with %d GPUs (%d inference + %d train)",
        resolved["num_gpus"],
        resolved["num_inference_gpus"],
        resolved["num_train_gpus"],
    )
    launch_training(resolved)


if __name__ == "__main__":
    main()
