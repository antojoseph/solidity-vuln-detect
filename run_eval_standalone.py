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
from openai import AsyncOpenAI

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
# Shared system prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(protocol_type: str) -> str:
    """Build the auditor system prompt for a scenario."""
    return (
        f"You are a smart contract security auditor reviewing a {protocol_type} protocol.\n\n"
        f"Instructions:\n"
        f"1. Use read_code() to examine the Solidity code snippet\n"
        f"2. Optionally use get_context() for protocol details\n"
        f"3. Optionally use list_hints() for detection guidance (reduces max reward)\n"
        f"4. Use submit_finding() with your analysis when ready\n\n"
        f"Your goal: identify the specific vulnerability type, explain the attack vector, "
        f"and pinpoint the affected lines of code."
    )


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
            "'oracle-manipulation', 'access-control', 'flash-loan', "
            "'precision-rounding', 'denial-of-service', 'frontrunning-mev', "
            "'integer-overflow', 'input-validation', 'reward-accounting', "
            "'slippage-protection', 'initialization', 'locked-funds', "
            "'governance', 'liquidation', 'stale-state', 'signature-replay', "
            "'incorrect-math', 'no-vulnerability', etc.)\n"
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

# OpenAI / OpenAI-compatible tool definitions
TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOL_DEFINITIONS
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
# Qwen native tool-call parsing (for servers without tool_call_parser)
# ---------------------------------------------------------------------------

def _parse_qwen_tool_calls(content: str) -> list[dict]:
    """Parse tool calls from Qwen's native text format.

    Supports both formats:
      Qwen3.5 XML:  <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
      Qwen3 JSON:   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    calls = []
    for m in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        block = m.group(1).strip()

        # Try Qwen3.5 XML format: <function=name><parameter=k>v</parameter>...</function>
        func_match = re.search(r"<function=(\w+)>(.*?)</function>", block, re.DOTALL)
        if func_match:
            name = func_match.group(1)
            params_block = func_match.group(2)
            args = {}
            for pm in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", params_block, re.DOTALL
            ):
                args[pm.group(1)] = pm.group(2).strip()
            calls.append({"name": name, "arguments": args})
            continue

        # Try JSON format: {"name": "...", "arguments": {...}}
        try:
            data = json.loads(block)
            calls.append({
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return calls


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
        # OpenAI-compatible endpoint (vLLM, SGLang, Ollama, etc.)
        api_key = os.environ.get("OPENAI_API_KEY", "empty")
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        episode_fn = run_episode_openai
        print(f"Provider: OpenAI-compatible ({base_url})")
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY not set. Add it to .env or export it.")
            sys.exit(1)
        client = AsyncAnthropic(api_key=api_key)
        episode_fn = run_episode
        print(f"Provider: Anthropic")

    # Load previous results for resume
    prev_results: dict[str, dict] = {}
    if resume_from:
        with open(resume_from) as f:
            prev_data = json.load(f)
        for trace in prev_data.get("traces", []):
            if trace.get("error") is None:
                prev_results[trace["scenario_id"]] = trace
        print(f"Resuming: {len(prev_results)} completed scenarios from {resume_from}")

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
            if s["id"] in prev_results:
                cached_results.append(prev_results[s["id"]])
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
