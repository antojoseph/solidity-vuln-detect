"""Standalone evaluation harness — no HUD credits required.

Calls the Anthropic or OpenAI-compatible API and runs local scoring from env.py.
Saves full message traces (RL-ready) and per-model result summaries.

Examples:
    # Anthropic models (default)
    python run_eval_standalone.py --model claude-opus-4-6 --max-tasks 1
    python run_eval_standalone.py --models claude-opus-4-6 claude-sonnet-4-6

    # OpenAI-compatible endpoint (e.g. local Qwen via vLLM/SGLang)
    python run_eval_standalone.py --base-url http://localhost:30000/v1 --model qwen3-32b --max-tasks 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from audit_core import (
    TOOL_DEFINITIONS,
    TOOL_DEFINITIONS_OPENAI,
    Episode,
    _parse_qwen_tool_calls,
    build_system_prompt,
)
from env import _load_scenarios

load_dotenv()


def _provider_key(base_url: str | None) -> str:
    return f"openai:{base_url}" if base_url else "anthropic"


def _resume_key(model: str, scenario_id: str, provider: str) -> str:
    return f"{provider}|{model}|{scenario_id}"


# ---------------------------------------------------------------------------
# Agent loop — single episode
# ---------------------------------------------------------------------------

async def run_episode(
    client: AsyncAnthropic,
    model: str,
    scenario: dict,
    max_steps: int = 10,
) -> dict:
    """Run a single eval episode. Returns trace dict."""
    ep = Episode(scenario)
    system_prompt = build_system_prompt(scenario["protocol_type"])

    messages = []
    steps = 0
    error = None

    try:
        for step in range(max_steps):
            steps = step + 1

            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                tools=TOOL_DEFINITIONS,
                messages=messages or [{"role": "user", "content": "Begin your audit."}],
            )

            # First iteration: add the initial user message to trace
            if step == 0:
                messages.append({"role": "user", "content": "Begin your audit."})

            # Build assistant message from response
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, agent is done
            if response.stop_reason == "end_turn":
                break

            # Dispatch tool calls
            tool_results = []
            submitted = False
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = ep.call_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
                if block.name == "submit_finding":
                    submitted = True

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if submitted:
                break

    except Exception as e:
        error = str(e)

    # Score
    result = ep.evaluate()
    result["messages"] = messages
    result["steps"] = steps
    result["error"] = error
    result["model"] = model
    result["scenario_id"] = scenario["id"]

    return result


# ---------------------------------------------------------------------------
# Agent loop — OpenAI-compatible endpoint (vLLM, SGLang, etc.)
# ---------------------------------------------------------------------------

async def run_episode_openai(
    client: AsyncOpenAI,
    model: str,
    scenario: dict,
    max_steps: int = 10,
) -> dict:
    """Run a single eval episode via an OpenAI-compatible API.

    Handles both structured tool_calls (server-parsed) and Qwen's native
    text-based tool calling format (parsed from content).
    """
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
                max_tokens=16384,
                temperature=0.7,
                top_p=0.8,
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
            reasoning = getattr(msg, "reasoning_content", None) or ""

            # --- Path A: structured tool_calls from the API ---
            if msg.tool_calls:
                assistant_msg = {"role": "assistant", "content": content}
                if reasoning:
                    assistant_msg["reasoning"] = reasoning
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

            # --- Path B: parse tool calls from content text (Qwen native) ---
            tool_calls = _parse_qwen_tool_calls(content)
            assistant_msg = {"role": "assistant", "content": content}
            if reasoning:
                assistant_msg["reasoning"] = reasoning
            messages.append(assistant_msg)

            if not tool_calls:
                break  # No tool calls, agent is done

            submitted = False
            tool_results = []
            for tc in tool_calls:
                result_text = ep.call_tool(tc["name"], tc["arguments"])
                tool_results.append(
                    f'{{"name": "{tc["name"]}", "content": {json.dumps(result_text)}}}'
                )
                if tc["name"] == "submit_finding":
                    submitted = True

            # Feed tool results back as a user message in Qwen's expected format
            results_text = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>" for r in tool_results
            )
            messages.append({"role": "user", "content": results_text})

            if submitted:
                break

    except Exception as e:
        error = str(e)

    # Score
    result = ep.evaluate()
    result["messages"] = messages
    result["steps"] = steps
    result["error"] = error
    result["model"] = model
    result["scenario_id"] = scenario["id"]

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

async def run_eval(
    models: list[str],
    task_file: str = "data/tasks_eval.json",
    max_tasks: int = 0,
    concurrency: int = 10,
    max_steps: int = 10,
    base_url: str | None = None,
    resume_from: str | None = None,
):
    """Run evaluation across models and tasks."""
    use_openai = base_url is not None

    if use_openai:
        if AsyncOpenAI is None:
            print("Error: openai is not installed. Install it or omit --base-url.")
            sys.exit(1)
        # OpenAI-compatible endpoint (vLLM, SGLang, Ollama, etc.)
        api_key = os.environ.get("OPENAI_API_KEY", "empty")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        episode_fn = run_episode_openai
        print(f"Provider: OpenAI-compatible ({base_url})")
    else:
        if AsyncAnthropic is None:
            print("Error: anthropic is not installed. Install it or use --base-url.")
            sys.exit(1)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
        client = AsyncAnthropic(api_key=api_key)
        episode_fn = run_episode
        print(f"Provider: Anthropic")

    provider = _provider_key(base_url)

    # Load previous results for resume
    prev_results: dict[str, dict] = {}
    if resume_from:
        with open(resume_from) as f:
            prev_data = json.load(f)
        for trace in prev_data.get("traces", []):
            if trace.get("error") is not None:
                continue
            key = _resume_key(
                trace.get("model", ""),
                trace.get("scenario_id", ""),
                trace.get("provider", prev_data.get("provider", "unknown")),
            )
            prev_results[key] = trace
        print(f"Resuming: {len(prev_results)} completed episodes from {resume_from}")

    # Load tasks
    with open(task_file) as f:
        tasks = json.load(f)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    # Load scenarios index
    all_scenarios = _load_scenarios()
    scenario_map = {s["id"]: s for s in all_scenarios}

    # Resolve scenario data for each task
    task_scenarios = []
    for t in tasks:
        sid = t.get("args", {}).get("scenario_id", "")
        if sid not in scenario_map:
            print(f"  Warning: scenario_id '{sid}' not found, skipping")
            continue
        task_scenarios.append(scenario_map[sid])

    print(f"Tasks: {len(task_scenarios)}")
    print(f"Models: {models}")
    print(f"Total episodes: {len(task_scenarios) * len(models)}")
    print(f"Concurrency: {concurrency}")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    for model in models:
        # Split into cached (from resume) and pending scenarios
        cached_results = []
        pending_scenarios = []
        for s in task_scenarios:
            key = _resume_key(model, s["id"], provider)
            if key in prev_results:
                cached_results.append(prev_results[key])
            else:
                pending_scenarios.append(s)

        total = len(pending_scenarios)
        completed = 0

        print(f"=== {model} ===")
        if cached_results:
            print(f"  Reusing {len(cached_results)} results from previous run")
        print(f"  Running {total} remaining episodes\n")

        async def run_with_semaphore(model: str, scenario: dict) -> dict:
            nonlocal completed
            async with semaphore:
                result = await episode_fn(client, model, scenario, max_steps=max_steps)
                result["provider"] = provider
                completed += 1
                reward = result["reward"]
                sid = result["scenario_id"]
                status = f"E {result['error'][:40]}" if result["error"] else f"R={reward:.3f}"
                print(f"  [{completed}/{total}] {model} | {sid} | {status}")
                return result

        coros = [run_with_semaphore(model, s) for s in pending_scenarios]
        new_results = await asyncio.gather(*coros)

        results = cached_results + list(new_results)

        # Aggregate
        rewards = [r["reward"] for r in results if r["error"] is None]
        errors = [r for r in results if r["error"] is not None]

        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        for r in results:
            if r["error"] is None:
                cat = r["info"].get("canonical_category", "unknown")
                diff = r["info"].get("difficulty", "unknown")
                by_category[cat].append(r["reward"])
                by_difficulty[diff].append(r["reward"])

        summary = {
            "model": model,
            "provider": provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_file": task_file,
            "total_tasks": len(task_scenarios),
            "completed": len(rewards),
            "errors": len(errors),
            "mean_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
            "median_reward": round(statistics.median(rewards), 4) if rewards else 0.0,
            "stdev_reward": round(statistics.stdev(rewards), 4) if len(rewards) > 1 else 0.0,
            "by_category": {
                k: round(statistics.mean(v), 4) for k, v in sorted(by_category.items())
            },
            "by_difficulty": {
                k: round(statistics.mean(v), 4) for k, v in sorted(by_difficulty.items())
            },
            "traces": results,
        }

        # Write results
        safe_model = model.replace("/", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"{safe_model}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n--- {model} Summary ---")
        print(f"  Completed: {len(rewards)}/{len(task_scenarios)}")
        print(f"  Errors: {len(errors)}")
        if rewards:
            print(f"  Mean reward: {summary['mean_reward']:.4f} ± {summary['stdev_reward']:.4f}")
            print(f"  Median reward: {summary['median_reward']:.4f}")
        print(f"  Results saved to: {out_path}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone eval harness (no HUD)")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (e.g. http://localhost:30000/v1)")
    parser.add_argument("--task-file", default="data/tasks_eval.json")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--resume", default=None,
                        help="Resume from a previous results file, skipping completed scenarios")
    args = parser.parse_args()

    models = args.models or [args.model]
    asyncio.run(run_eval(
        models=models,
        task_file=args.task_file,
        max_tasks=args.max_tasks,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
        base_url=args.base_url,
        resume_from=args.resume,
    ))


if __name__ == "__main__":
    main()
