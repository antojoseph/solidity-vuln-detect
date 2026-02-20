"""Batch evaluation harness for the Solidity vulnerability detection environment.

Modes:
    python run_eval.py --mode quick       Single random episode (smoke test)
    python run_eval.py --mode eval        Full eval suite across models
    python run_eval.py --mode train       Training data collection (many episodes)
    python run_eval.py --mode checkpoint  Evaluate a fine-tuned checkpoint

Examples:
    python run_eval.py --mode quick
    python run_eval.py --mode eval --models gpt-4o-mini gpt-4o
    python run_eval.py --mode train --group 3
    python run_eval.py --mode checkpoint --checkpoint my-org/vuln-detect-v1
"""

import argparse
import asyncio
import sys

import hud
from hud.agents import OpenAIChatAgent
from hud.datasets import load_tasks

from env import env


# ---------------------------------------------------------------------------
# Mode 1: Quick test (single episode)
# ---------------------------------------------------------------------------

async def run_quick(model: str = "gpt-4o-mini"):
    """Run a single random episode to verify the environment works."""
    print(f"=== Quick Test (model={model}) ===\n")

    task = env("detect-vuln")

    async with hud.eval(task) as ctx:
        agent = OpenAIChatAgent.create(model=model)
        await agent.run(ctx, max_steps=10)

    print(f"\nReward: {ctx.reward}")


# ---------------------------------------------------------------------------
# Mode 2: Full eval suite
# ---------------------------------------------------------------------------

async def run_eval(models: list[str], max_tasks: int = 0):
    """Run the held-out eval set across one or more models."""
    print(f"=== Eval Suite (models={models}) ===\n")

    all_tasks = load_tasks("data/tasks_eval.json")
    bound = [env(t.scenario, **t.args) for t in all_tasks]

    if max_tasks > 0:
        bound = bound[:max_tasks]

    print(f"Tasks: {len(bound)}")
    print(f"Models: {models}")

    if len(models) > 1:
        async with hud.eval(bound, variants={"model": models}) as ctx:
            agent = OpenAIChatAgent.create(model=ctx.variants["model"])
            await agent.run(ctx, max_steps=10)
    else:
        async with hud.eval(bound) as ctx:
            agent = OpenAIChatAgent.create(model=models[0])
            await agent.run(ctx, max_steps=10)

    print("\nEval complete. Check hud.ai for full results.")


# ---------------------------------------------------------------------------
# Mode 3: Training data collection
# ---------------------------------------------------------------------------

async def run_train(model: str = "gpt-4o-mini", group: int = 3, max_tasks: int = 0):
    """Run the training set to collect trajectories for RL training."""
    print(f"=== Training Data Collection (model={model}, group={group}) ===\n")

    all_tasks = load_tasks("data/tasks_train.json")
    bound = [env(t.scenario, **t.args) for t in all_tasks]

    if max_tasks > 0:
        bound = bound[:max_tasks]

    print(f"Tasks: {len(bound)}")
    print(f"Episodes per task: {group}")
    print(f"Total episodes: {len(bound) * group}")

    async with hud.eval(bound, group=group) as ctx:
        agent = OpenAIChatAgent.create(model=model)
        await agent.run(ctx, max_steps=10)

    print("\nTraining data collected. Check hud.ai for trajectories.")


# ---------------------------------------------------------------------------
# Mode 4: Checkpoint evaluation
# ---------------------------------------------------------------------------

async def run_checkpoint(checkpoint: str, model: str = "gpt-4o-mini", max_tasks: int = 0):
    """Evaluate a fine-tuned checkpoint against the eval set."""
    print(f"=== Checkpoint Eval (checkpoint={checkpoint}, base={model}) ===\n")

    all_tasks = load_tasks("data/tasks_eval.json")
    bound = [env(t.scenario, **t.args) for t in all_tasks]

    if max_tasks > 0:
        bound = bound[:max_tasks]

    print(f"Tasks: {len(bound)}")

    async with hud.eval(bound) as ctx:
        agent = OpenAIChatAgent.create(model=model, checkpoint=checkpoint)
        await agent.run(ctx, max_steps=10)

    print("\nCheckpoint eval complete. Check hud.ai for results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Solidity vulnerability detection eval harness")
    parser.add_argument("--mode", choices=["quick", "eval", "train", "checkpoint"], default="quick")
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--group", type=int, default=3)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    args = parser.parse_args()

    if args.mode == "quick":
        asyncio.run(run_quick(args.model))
    elif args.mode == "eval":
        asyncio.run(run_eval(args.models, args.max_tasks))
    elif args.mode == "train":
        asyncio.run(run_train(args.model, args.group, args.max_tasks))
    elif args.mode == "checkpoint":
        if not args.checkpoint:
            print("Error: --checkpoint is required for checkpoint mode")
            sys.exit(1)
        asyncio.run(run_checkpoint(args.checkpoint, args.model, args.max_tasks))


if __name__ == "__main__":
    main()
