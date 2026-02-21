"""Batch evaluation harness for the Solidity vulnerability detection environment.

Modes:
    python run_eval.py --mode quick       Single random episode (smoke test)
    python run_eval.py --mode eval        Full eval suite across models
    python run_eval.py --mode train       Training data collection (many episodes)
    python run_eval.py --mode curriculum  Curriculum training (easy → hard)
    python run_eval.py --mode checkpoint  Evaluate a fine-tuned checkpoint

Examples:
    python run_eval.py --mode quick
    python run_eval.py --mode eval --models gpt-4o-mini gpt-4o
    python run_eval.py --mode eval --models gpt-4o-mini --report-clean
    python run_eval.py --mode eval --models gpt-4o-mini --report-ood
    python run_eval.py --mode train --group 3
    python run_eval.py --mode curriculum --group 3
    python run_eval.py --mode checkpoint --checkpoint my-org/vuln-detect-v1
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

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

async def _run_eval_on_tasks(task_path: str, models: list[str], max_tasks: int, label: str):
    all_tasks = load_tasks(task_path)
    bound = [env(t.scenario, **t.args) for t in all_tasks]

    if max_tasks > 0:
        bound = bound[:max_tasks]

    print(f"=== {label} (models={models}) ===\n")
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


async def run_eval(models: list[str], max_tasks: int = 0, report_clean: bool = False, report_ood: bool = False):
    """Run the held-out eval set across one or more models."""
    await _run_eval_on_tasks("data/tasks_eval.json", models, max_tasks, "Eval Suite")

    if report_clean:
        clean_path = "data/tasks_eval_clean.json"
        if not Path(clean_path).exists():
            print(f"\nSkipped clean report: {clean_path} not found.")
        else:
            await _run_eval_on_tasks(clean_path, models, max_tasks, "Clean Eval Suite")

    if report_ood:
        ood_path = "data/tasks_eval_ood.json"
        if not Path(ood_path).exists():
            print(f"\nSkipped OOD report: {ood_path} not found.")
        else:
            await _run_eval_on_tasks(ood_path, models, max_tasks, "OOD Eval Suite")


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
# Mode 4: Curriculum training
# ---------------------------------------------------------------------------

async def run_curriculum(model: str = "gpt-4o-mini", group: int = 3, max_tasks: int = 0):
    """Run curriculum training ordered by difficulty (easy → hard)."""
    print(f"=== Curriculum Training (model={model}, group={group}) ===\n")

    all_tasks = load_tasks("data/tasks_train.json")
    scenarios = json.loads(Path("data/scenarios.json").read_text(encoding="utf-8"))
    difficulty_map = {s["id"]: s.get("difficulty", "unknown") for s in scenarios}
    order = {"easy": 0, "medium": 1, "hard": 2}

    sorted_tasks = sorted(
        all_tasks,
        key=lambda t: (order.get(difficulty_map.get(t.args.get("scenario_id"), "unknown"), 99)),
    )
    bound = [env(t.scenario, **t.args) for t in sorted_tasks]

    if max_tasks > 0:
        bound = bound[:max_tasks]

    print(f"Tasks: {len(bound)}")
    print(f"Episodes per task: {group}")
    print(f"Total episodes: {len(bound) * group}")

    async with hud.eval(bound, group=group) as ctx:
        agent = OpenAIChatAgent.create(model=model)
        await agent.run(ctx, max_steps=10)

    print("\nCurriculum data collected. Check hud.ai for trajectories.")


# ---------------------------------------------------------------------------
# Mode 5: Checkpoint evaluation
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
    parser.add_argument(
        "--mode",
        choices=["quick", "eval", "train", "curriculum", "checkpoint"],
        default="quick",
    )
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--group", type=int, default=3)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument("--report-clean", action="store_true", help="Eval clean-only tasks")
    parser.add_argument("--report-ood", action="store_true", help="Eval OOD tasks")
    args = parser.parse_args()

    if args.mode == "quick":
        asyncio.run(run_quick(args.model))
    elif args.mode == "eval":
        asyncio.run(run_eval(args.models, args.max_tasks, args.report_clean, args.report_ood))
    elif args.mode == "train":
        asyncio.run(run_train(args.model, args.group, args.max_tasks))
    elif args.mode == "curriculum":
        asyncio.run(run_curriculum(args.model, args.group, args.max_tasks))
    elif args.mode == "checkpoint":
        if not args.checkpoint:
            print("Error: --checkpoint is required for checkpoint mode")
            sys.exit(1)
        asyncio.run(run_checkpoint(args.checkpoint, args.model, args.max_tasks))


if __name__ == "__main__":
    main()
