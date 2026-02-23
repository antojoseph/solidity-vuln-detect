"""Standalone evaluation harness — no HUD credits required.

Calls the Anthropic API directly and runs local scoring from env.py.
Saves full message traces (RL-ready) and per-model result summaries.

Examples:
    python run_eval_standalone.py --model claude-opus-4-6 --max-tasks 1
    python run_eval_standalone.py --models claude-opus-4-6 claude-sonnet-4-6
    python run_eval_standalone.py --model claude-opus-4-6 --concurrency 20
    python run_eval_standalone.py --model claude-opus-4-6 --task-file data/tasks_eval_ood.json
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

# Import scoring functions from env.py (all pure functions, no globals)
from env import (
    CATEGORY_SEVERITY,
    _load_scenarios,
    _score_category,
    _score_explanation,
    _score_exploitability,
    _score_lines,
    _score_severity,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Anthropic tool definitions (mirror env.py function signatures)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "read_code",
        "description": (
            "Read the Solidity code snippet under review. "
            "Returns the smart contract code that may contain a security vulnerability. "
            "Examine it carefully for common vulnerability patterns like reentrancy, "
            "access control issues, oracle manipulation, etc."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_context",
        "description": (
            "Get context about the protocol being audited. "
            "Returns the protocol type and preconditions that describe when this "
            "vulnerability pattern applies. No reward penalty for using this tool."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_hints",
        "description": (
            "Request detection heuristics for analyzing this code. "
            "Returns a list of things to look for when auditing this code. "
            "WARNING: Using hints reduces your maximum reward from 1.0 to 0.7."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_finding",
        "description": (
            "Submit your vulnerability analysis.\n\n"
            "Args:\n"
            "    vulnerability_type: Category of vulnerability (e.g. 'reentrancy', "
            "'oracle manipulation', 'access control', 'flash loan', "
            "'precision loss', 'denial of service', 'frontrunning', etc.)\n"
            "    explanation: Detailed explanation of WHY the code is vulnerable, "
            "including what an attacker could exploit and how.\n"
            "    severity: Impact level — 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', "
            "or 'NONE' for safe code.\n"
            "    affected_lines: Comma-separated line numbers where the root cause is "
            "(e.g. '5,6,7'). Use the line numbers from read_code().\n"
            "    attack_path: Step-by-step outline of how the exploit is executed.\n"
            "    prerequisites: Preconditions required for the exploit to work.\n"
            "    impact: Expected impact if exploited (e.g. fund loss, stolen assets)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vulnerability_type": {
                    "type": "string",
                    "description": "Category of vulnerability",
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of the vulnerability",
                },
                "severity": {
                    "type": "string",
                    "description": "Impact level: CRITICAL, HIGH, MEDIUM, LOW, or NONE",
                    "default": "HIGH",
                },
                "affected_lines": {
                    "type": "string",
                    "description": "Comma-separated line numbers of root cause",
                    "default": "",
                },
                "attack_path": {
                    "type": "string",
                    "description": "Step-by-step exploit outline",
                    "default": "",
                },
                "prerequisites": {
                    "type": "string",
                    "description": "Preconditions for the exploit",
                    "default": "",
                },
                "impact": {
                    "type": "string",
                    "description": "Expected impact if exploited",
                    "default": "",
                },
            },
            "required": ["vulnerability_type", "explanation"],
        },
    },
]


# ---------------------------------------------------------------------------
# Per-episode state (thread-safe via asyncio — no shared mutation)
# ---------------------------------------------------------------------------

class Episode:
    """Encapsulates all state for a single eval episode."""

    def __init__(self, scenario: dict):
        self.scenario = scenario.copy()
        self.hints_used = False
        self.code_read = False
        self.submission: dict | None = None

    def read_code(self) -> str:
        self.code_read = True
        code = self.scenario.get("code_clean", "No code loaded.")
        lines = code.split("\n")
        return "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))

    def get_context(self) -> str:
        protocol_type = self.scenario.get("protocol_type", "unknown")
        protocol_name = self.scenario.get("protocol_name", "")
        preconditions = self.scenario.get("preconditions", "No preconditions available.")
        header = f"Protocol type: {protocol_type}"
        if protocol_name:
            header += f"\nProtocol: {protocol_name}"
        return f"{header}\n\nPreconditions:\n{preconditions}"

    def list_hints(self) -> str:
        self.hints_used = True
        hints = self.scenario.get("hints", [])
        if not hints:
            return "No hints available for this snippet."
        return "Detection heuristics:\n" + "\n".join(
            f"  {i+1}. {h}" for i, h in enumerate(hints)
        )

    def submit_finding(
        self,
        vulnerability_type: str,
        explanation: str,
        severity: str = "HIGH",
        affected_lines: str = "",
        attack_path: str = "",
        prerequisites: str = "",
        impact: str = "",
    ) -> str:
        parsed_lines = []
        if affected_lines.strip():
            for part in affected_lines.split(","):
                part = part.strip()
                if part.isdigit():
                    parsed_lines.append(int(part))

        self.submission = {
            "vulnerability_type": vulnerability_type,
            "explanation": explanation,
            "severity": severity.upper(),
            "affected_lines": parsed_lines,
            "attack_path": attack_path,
            "prerequisites": prerequisites,
            "impact": impact,
        }
        return f"Finding submitted: {vulnerability_type} ({severity}). Awaiting evaluation."

    def call_tool(self, name: str, args: dict) -> str:
        if name == "read_code":
            return self.read_code()
        elif name == "get_context":
            return self.get_context()
        elif name == "list_hints":
            return self.list_hints()
        elif name == "submit_finding":
            return self.submit_finding(**args)
        else:
            return f"Unknown tool: {name}"

    def evaluate(self) -> dict:
        """Run deterministic scoring. Returns result dict with reward + subscores."""
        if not self.submission:
            return {
                "reward": 0.0,
                "subscores": {},
                "info": {"scenario_id": self.scenario["id"], "error": "no_submission"},
            }

        cat_score = _score_category(
            self.submission["vulnerability_type"],
            self.scenario["category_slug"],
            self.scenario["canonical_category"],
        )
        expl_score = _score_explanation(self.submission["explanation"], self.scenario)
        sev_score = _score_severity(self.submission["severity"], self.scenario)
        line_score = _score_lines(
            self.submission["affected_lines"],
            self.scenario.get("bug_lines", []),
        )
        exploit_score = _score_exploitability(
            self.submission.get("attack_path", ""),
            self.submission.get("prerequisites", ""),
            self.submission.get("impact", ""),
            ground_canonical=self.scenario["canonical_category"],
        )

        # Schema penalty
        schema_penalty = 1.0
        if self.scenario["canonical_category"] != "no-vulnerability":
            if all(
                not self.submission.get(f, "").strip()
                for f in ("attack_path", "prerequisites", "impact")
            ):
                schema_penalty *= 0.9
        if "," in self.submission["vulnerability_type"]:
            schema_penalty *= 0.85

        raw = (
            0.40 * cat_score
            + 0.25 * expl_score
            + 0.10 * sev_score
            + 0.15 * line_score
            + 0.10 * exploit_score
        )
        hint_penalty = 0.7 if self.hints_used else 1.0
        final = round(raw * schema_penalty * hint_penalty, 4)

        return {
            "reward": final,
            "subscores": {
                "category_match": round(cat_score, 4),
                "explanation_quality": round(expl_score, 4),
                "severity_match": round(sev_score, 4),
                "line_accuracy": round(line_score, 4),
                "exploitability": round(exploit_score, 4),
            },
            "info": {
                "scenario_id": self.scenario["id"],
                "canonical_category": self.scenario["canonical_category"],
                "difficulty": self.scenario.get("difficulty", "unknown"),
                "hints_used": self.hints_used,
                "code_read": self.code_read,
                "schema_penalty": schema_penalty,
                "submitted_type": self.submission["vulnerability_type"],
                "submitted_severity": self.submission["severity"],
                "submitted_lines": self.submission["affected_lines"],
                "ground_truth_lines": self.scenario.get("bug_lines", []),
            },
        }


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
    protocol_type = scenario["protocol_type"]

    system_prompt = (
        f"You are a smart contract security auditor reviewing a {protocol_type} protocol.\n\n"
        f"Instructions:\n"
        f"1. Use read_code() to examine the Solidity code snippet\n"
        f"2. Optionally use get_context() for protocol details\n"
        f"3. Optionally use list_hints() for detection guidance (reduces max reward)\n"
        f"4. Use submit_finding() with your analysis when ready\n\n"
        f"Your goal: identify the specific vulnerability type, explain the attack vector, "
        f"and pinpoint the affected lines of code."
    )

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
# Batch runner
# ---------------------------------------------------------------------------

async def run_eval(
    models: list[str],
    task_file: str = "data/tasks_eval.json",
    max_tasks: int = 0,
    concurrency: int = 10,
    max_steps: int = 10,
):
    """Run evaluation across models and tasks."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    client = AsyncAnthropic(api_key=api_key)

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
    total = len(task_scenarios) * len(models)

    async def run_with_semaphore(model: str, scenario: dict) -> dict:
        nonlocal completed
        async with semaphore:
            result = await run_episode(client, model, scenario, max_steps=max_steps)
            completed += 1
            reward = result["reward"]
            sid = result["scenario_id"]
            status = f"E {result['error'][:40]}" if result["error"] else f"R={reward:.3f}"
            print(f"  [{completed}/{total}] {model} | {sid} | {status}")
            return result

    for model in models:
        print(f"=== {model} ===\n")

        coros = [run_with_semaphore(model, s) for s in task_scenarios]
        results = await asyncio.gather(*coros)

        # Aggregate
        rewards = [r["reward"] for r in results if r["error"] is None]
        errors = [r for r in results if r["error"] is not None]

        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        for r in results:
            if r["error"] is None:
                by_category[r["info"]["canonical_category"]].append(r["reward"])
                by_difficulty[r["info"]["difficulty"]].append(r["reward"])

        summary = {
            "model": model,
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
    parser.add_argument("--task-file", default="data/tasks_eval.json")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    models = args.models or [args.model]
    asyncio.run(run_eval(
        models=models,
        task_file=args.task_file,
        max_tasks=args.max_tasks,
        concurrency=args.concurrency,
        max_steps=args.max_steps,
    ))


if __name__ == "__main__":
    main()
