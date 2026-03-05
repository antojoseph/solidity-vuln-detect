"""W&B monitor for Solidity vuln detection RL + eval pipelines.

Reads existing progress logs and trace files retroactively, then watches
for new data in real-time. Runs as a sidecar container alongside the RL job.

Usage:
    python3 wandb_monitor.py --rl-dir /workspace/rl --project solidity-vuln-detect
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import wandb


def parse_progress_line(line: str) -> dict | None:
    """Parse a progress.log line into a dict of metrics."""
    # Format: 2026-03-05T18:54:42+00:00 step=1 scenarios=32/3834 batch_reward=0.5188 ...
    m = re.search(
        r"step=(\d+)\s+scenarios=(\d+)/(\d+)\s+"
        r"batch_reward=([\d.]+)\s+running_reward=([\d.]+)\s+"
        r"batch_advantage=([+\-\d.]+)\s+errors=(\d+)\s+"
        r"elapsed=([\d.]+)s",
        line,
    )
    if not m:
        return None
    return {
        "step": int(m.group(1)),
        "scenarios_done": int(m.group(2)),
        "scenarios_total": int(m.group(3)),
        "batch_reward": float(m.group(4)),
        "running_reward": float(m.group(5)),
        "batch_advantage": float(m.group(6)),
        "errors": int(m.group(7)),
        "elapsed_s": float(m.group(8)),
    }


def parse_batch_traces(batch_file: Path) -> dict:
    """Parse a batch trace JSONL file and compute summary stats."""
    rewards = []
    categories = defaultdict(list)
    subscores = defaultdict(list)
    no_submission = 0

    with open(batch_file) as f:
        for line in f:
            trace = json.loads(line)
            r = trace.get("reward", 0)
            rewards.append(r)
            if r == 0.0:
                no_submission += 1

            cat = trace.get("info", {}).get("canonical_category", "unknown")
            categories[cat].append(r)

            for k, v in trace.get("subscores", {}).items():
                subscores[k].append(v)

    if not rewards:
        return {}

    stats = {
        "trace/mean_reward": sum(rewards) / len(rewards),
        "trace/min_reward": min(rewards),
        "trace/max_reward": max(rewards),
        "trace/no_submission_pct": no_submission / len(rewards) * 100,
        "trace/count": len(rewards),
    }

    for k, vals in subscores.items():
        stats[f"subscore/{k}"] = sum(vals) / len(vals)

    # Top/bottom categories
    cat_means = {c: sum(v) / len(v) for c, v in categories.items() if len(v) >= 2}
    if cat_means:
        best_cat = max(cat_means, key=cat_means.get)
        worst_cat = min(cat_means, key=cat_means.get)
        stats["category/best"] = cat_means[best_cat]
        stats["category/worst"] = cat_means[worst_cat]

    return stats


def load_eval_results(results_dir: Path) -> None:
    """Log eval result files as W&B summary tables."""
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.load(open(f))
            model = data.get("model", f.stem)
            mean_r = data.get("mean_reward", 0)
            median_r = data.get("median_reward", 0)
            completed = data.get("completed", 0)
            total = data.get("total_tasks", 0)

            wandb.log({
                f"eval/{model}/mean_reward": mean_r,
                f"eval/{model}/median_reward": median_r,
                f"eval/{model}/completed": completed,
                f"eval/{model}/total": total,
            }, commit=False)

            # Per-category table
            by_cat = data.get("by_category", {})
            if by_cat:
                table = wandb.Table(columns=["category", "mean_reward", "count"])
                for cat, stats in sorted(by_cat.items()):
                    if isinstance(stats, dict):
                        table.add_data(cat, stats.get("mean", 0), stats.get("count", 0))
                    elif isinstance(stats, list):
                        table.add_data(cat, sum(stats) / len(stats) if stats else 0, len(stats))
                wandb.log({f"eval/{model}/categories": table}, commit=False)

        except Exception as e:
            print(f"  Skipped {f.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="W&B monitor for RL + eval")
    parser.add_argument("--rl-dir", default="/workspace/rl")
    parser.add_argument("--results-dir", default="/workspace/results")
    parser.add_argument("--project", default="solidity-vuln-detect")
    parser.add_argument("--run-name", default="qwen35-9b-sft-rl")
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    rl_dir = Path(args.rl_dir)
    results_dir = Path(args.results_dir)
    progress_log = rl_dir / "progress.log"
    traces_dir = rl_dir / "traces"

    # Init W&B
    wandb.init(
        project=args.project,
        name=args.run_name,
        config={
            "model": "Qwen3.5-9B-SFT",
            "method": "RLOO",
            "samples_per_scenario": 6,
            "batch_size": 32,
            "training_scenarios": 3834,
        },
    )

    print(f"W&B run: {wandb.run.url}")
    print(f"Monitoring: {rl_dir}")

    # Log eval results (one-time)
    if results_dir.exists():
        print(f"Loading eval results from {results_dir}...")
        load_eval_results(results_dir)
        wandb.log({}, commit=True)

    # Track what we've already processed
    last_progress_line = 0
    processed_batches = set()

    while True:
        logged = False

        # Parse new progress log entries
        if progress_log.exists():
            with open(progress_log) as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if i < last_progress_line:
                    continue
                metrics = parse_progress_line(line)
                if metrics:
                    step = metrics.pop("step")
                    total = metrics.pop("scenarios_total")
                    metrics["progress_pct"] = metrics["scenarios_done"] / total * 100
                    metrics["throughput_scenarios_per_min"] = (
                        metrics["scenarios_done"] / (metrics["elapsed_s"] / 60)
                        if metrics["elapsed_s"] > 0 else 0
                    )
                    # Merge trace stats for this step if available
                    batch_file = traces_dir / f"batch_{step:04d}.jsonl"
                    if batch_file.exists() and batch_file.name not in processed_batches:
                        trace_stats = parse_batch_traces(batch_file)
                        metrics.update(trace_stats)
                        processed_batches.add(batch_file.name)
                    wandb.log(metrics, step=step)
                    logged = True
            last_progress_line = len(lines)

        # Parse any remaining batch trace files not matched to progress lines
        if traces_dir.exists():
            for batch_file in sorted(traces_dir.glob("batch_*.jsonl")):
                if batch_file.name in processed_batches:
                    continue
                stats = parse_batch_traces(batch_file)
                if stats:
                    step_match = re.search(r"batch_(\d+)", batch_file.name)
                    step = int(step_match.group(1)) if step_match else 0
                    wandb.log(stats, step=step)
                    logged = True
                processed_batches.add(batch_file.name)

        if logged:
            print(f"  Logged data up to step {last_progress_line} "
                  f"({len(processed_batches)} batches)")

        # Check if RL is done
        summary_file = rl_dir / "summary.json"
        if summary_file.exists() and summary_file.name not in processed_batches:
            summary = json.load(open(summary_file))
            wandb.summary.update(summary)
            print(f"RL complete! Final summary: {summary}")
            processed_batches.add(summary_file.name)
            break

        time.sleep(args.poll_interval)

    wandb.finish()
    print("W&B monitoring complete.")


if __name__ == "__main__":
    main()
