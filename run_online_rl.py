"""Online rollout collection for Solidity vulnerability detection.

This script samples multi-attempt audit rollouts, scores them with the live
environment, and computes RLOO-style advantages for analysis. It does not
perform optimizer steps or update model weights. For real online RL training,
use run_skyrl_train.py.

Architecture:
    vLLM server (GPU 0) ←→ This script (GPU 1) ←→ SolidityVulnEnv (CPU)

    For each batch:
    1. Sample N scenarios from training data
    2. For each scenario, generate K independent audit attempts via vLLM
    3. Score each attempt with the deterministic scorer
    4. Compute RLOO advantages: advantage_i = reward_i - mean(other rewards)
    5. Log results, save best traces

Usage:
    python3 run_online_rl.py --model /path/to/sft-merged --data data/skyrl_prompts.jsonl \
        --output outputs/rl --vllm-url http://localhost:30002/v1

Designed to survive session drops when run via Docker (see run_online_rl.sh).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

try:
    import yaml
except ImportError:
    yaml = None

from audit_core import (
    Episode,
    TOOL_DEFINITIONS_OPENAI,
    build_system_prompt,
    _parse_qwen_tool_calls,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "scenarios.json"


def load_scenarios() -> dict[str, dict]:
    with open(DATA_PATH) as f:
        return {s["id"]: s for s in json.load(f)}




def _load_rollout_defaults(config_path: str) -> dict:
    if not config_path or yaml is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    online_rl = config.get("online_rl", {})
    return {
        "samples_per_scenario": int(online_rl.get("num_generations", 6)),
        "batch_size": int(online_rl.get("batch_size", 32)),
        "max_steps": int(online_rl.get("max_tool_calling_iterations", 5)),
        "temperature": float(online_rl.get("generation_temperature", 0.7)),
        "concurrency": int(online_rl.get("max_concurrent_requests", 64)),
    }


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------

async def run_rollout(
    client: AsyncOpenAI,
    model: str,
    scenario: dict,
    max_steps: int = 5,
    temperature: float = 0.7,
) -> dict:
    """Run one audit attempt against a scenario. Returns reward + trace."""
    ep = Episode(scenario)
    system_prompt = build_system_prompt(scenario["protocol_type"])

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Begin your audit."},
    ]
    steps = 0
    error = None

    try:
        for step in range(max_steps):
            steps = step + 1
            response = await client.chat.completions.create(
                model=model,
                max_tokens=8192,
                temperature=temperature,
                top_p=0.95,
                presence_penalty=1.5,
                messages=messages,
                tools=TOOL_DEFINITIONS_OPENAI,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""

            # Structured tool calls from vLLM
            if msg.tool_calls:
                assistant_msg = {"role": "assistant", "content": content}
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
                messages.append(assistant_msg)

                submitted = False
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result_text = ep.call_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
                    if tc.function.name == "submit_finding":
                        submitted = True
                if submitted:
                    break
                continue

            # Text-based tool calls (fallback)
            tool_calls = _parse_qwen_tool_calls(content)
            messages.append({"role": "assistant", "content": content})
            if not tool_calls:
                break
            submitted = False
            results = []
            for tc in tool_calls:
                result_text = ep.call_tool(tc["name"], tc["arguments"])
                results.append(f'{{"name": "{tc["name"]}", "content": {json.dumps(result_text)}}}')
                if tc["name"] == "submit_finding":
                    submitted = True
            messages.append({
                "role": "user",
                "content": "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in results),
            })
            if submitted:
                break

    except Exception as e:
        error = str(e)

    result = ep.evaluate()
    return {
        "reward": result["reward"],
        "subscores": result.get("subscores", {}),
        "info": result.get("info", {}),
        "scenario_id": scenario["id"],
        "steps": steps,
        "error": error,
    }


# ---------------------------------------------------------------------------
# RLOO batch
# ---------------------------------------------------------------------------

async def run_rloo_batch(
    client: AsyncOpenAI,
    model: str,
    scenarios: list[dict],
    samples_per_scenario: int = 6,
    max_steps: int = 5,
    temperature: float = 0.7,
    semaphore: asyncio.Semaphore | None = None,
) -> list[dict]:
    """Run RLOO batch: K independent rollouts per scenario.

    Returns list of dicts with reward, rloo_advantage, scenario_id.
    """
    sem = semaphore or asyncio.Semaphore(16)

    async def _run(scenario):
        async with sem:
            return await run_rollout(client, model, scenario, max_steps, temperature)

    # Generate K rollouts per scenario
    all_results = []
    for scenario in scenarios:
        coros = [_run(scenario) for _ in range(samples_per_scenario)]
        group_results = await asyncio.gather(*coros)

        # RLOO advantage: reward_i - mean(rewards_j for j != i)
        rewards = [r["reward"] for r in group_results]
        for i, r in enumerate(group_results):
            others = [rewards[j] for j in range(len(rewards)) if j != i]
            baseline = sum(others) / len(others) if others else 0
            r["rloo_advantage"] = r["reward"] - baseline
            r["group_mean"] = sum(rewards) / len(rewards)
            r["group_std"] = statistics.stdev(rewards) if len(rewards) > 1 else 0
        all_results.extend(group_results)

    return all_results


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

async def train(
    vllm_url: str,
    model: str,
    prompts: list[dict],
    scenario_map: dict[str, dict],
    output_dir: Path,
    max_scenarios: int = 0,
    samples_per_scenario: int = 6,
    batch_size: int = 32,
    max_steps: int = 5,
    temperature: float = 0.7,
    concurrency: int = 64,
):
    client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY", timeout=600)
    progress_log = output_dir / "progress.log"
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Filter to valid scenarios
    valid_prompts = [p for p in prompts if p["extras"]["scenario_id"] in scenario_map]
    if max_scenarios > 0:
        valid_prompts = valid_prompts[:max_scenarios]

    logger.info(f"Collecting rollouts for {len(valid_prompts)} scenarios, "
                f"{samples_per_scenario} samples each, batch_size={batch_size}")

    # Shuffle
    random.shuffle(valid_prompts)

    # Stats
    all_rewards = []
    all_advantages = []
    step = 0
    sem = asyncio.Semaphore(concurrency)

    for batch_start in range(0, len(valid_prompts), batch_size):
        batch_prompts = valid_prompts[batch_start:batch_start + batch_size]
        batch_scenarios = [
            scenario_map[p["extras"]["scenario_id"]] for p in batch_prompts
        ]

        step += 1
        t0 = time.time()

        results = await run_rloo_batch(
            client, model, batch_scenarios,
            samples_per_scenario=samples_per_scenario,
            max_steps=max_steps,
            temperature=temperature,
            semaphore=sem,
        )

        elapsed = time.time() - t0
        batch_rewards = [r["reward"] for r in results if not r.get("error")]
        batch_advantages = [r["rloo_advantage"] for r in results if not r.get("error")]
        all_rewards.extend(batch_rewards)
        all_advantages.extend(batch_advantages)

        mean_r = statistics.mean(batch_rewards) if batch_rewards else 0
        mean_adv = statistics.mean(batch_advantages) if batch_advantages else 0
        running_mean = statistics.mean(all_rewards) if all_rewards else 0
        errors = sum(1 for r in results if r.get("error"))

        log_line = (
            f"step={step} scenarios={batch_start+len(batch_prompts)}/{len(valid_prompts)} "
            f"batch_reward={mean_r:.4f} running_reward={running_mean:.4f} "
            f"batch_advantage={mean_adv:+.4f} errors={errors} "
            f"elapsed={elapsed:.1f}s"
        )
        logger.info(log_line)

        # Append to progress log (survives container restart)
        with open(progress_log, "a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()} {log_line}\n")

        # Save batch traces
        trace_file = traces_dir / f"batch_{step:04d}.jsonl"
        with open(trace_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Final summary
    summary = {
        "mode": "rollout_collection",
        "uses_weight_updates": False,
        "total_scenarios": len(valid_prompts),
        "samples_per_scenario": samples_per_scenario,
        "total_rollouts": len(all_rewards),
        "mean_reward": statistics.mean(all_rewards) if all_rewards else 0,
        "median_reward": statistics.median(all_rewards) if all_rewards else 0,
        "std_reward": statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0,
        "mean_advantage": statistics.mean(all_advantages) if all_advantages else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Rollout collection complete. Summary: {summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Online rollout collection for Solidity vuln detection")
    parser.add_argument("--model", default="Qwen3.5-9B-SFT",
                        help="Model name (as served by vLLM)")
    parser.add_argument("--data", default="data/skyrl_prompts.jsonl")
    parser.add_argument("--output", default="outputs/rl")
    parser.add_argument("--config", default="training.yaml")
    parser.add_argument("--vllm-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:30002/v1"))
    parser.add_argument("--max-scenarios", type=int, default=0,
                        help="Limit scenarios (0=all)")
    parser.add_argument("--samples-per-scenario", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=64,
                        help="Max concurrent API requests (match to vLLM replicas)")
    args = parser.parse_args()

    defaults = {
        "samples_per_scenario": 6,
        "batch_size": 32,
        "max_steps": 5,
        "temperature": 0.7,
        "concurrency": 64,
    }
    config_defaults = _load_rollout_defaults(args.config)
    for key, default_value in defaults.items():
        if key in config_defaults and getattr(args, key) == default_value:
            setattr(args, key, config_defaults[key])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_map = load_scenarios()
    prompts = load_prompts(args.data)

    logger.info(f"Loaded {len(scenario_map)} scenarios, {len(prompts)} prompts")
    logger.info(f"vLLM URL: {args.vllm_url}")
    logger.info(f"Output: {output_dir}")
    logger.info("Note: run_online_rl.py collects scored rollouts only; it does not update weights.")

    asyncio.run(train(
        vllm_url=args.vllm_url,
        model=args.model,
        prompts=prompts,
        scenario_map=scenario_map,
        output_dir=output_dir,
        max_scenarios=args.max_scenarios,
        samples_per_scenario=args.samples_per_scenario,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        temperature=args.temperature,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
