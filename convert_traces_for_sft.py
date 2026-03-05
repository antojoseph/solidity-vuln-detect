"""Convert Claude eval traces to SLIME SFT format.

Reads result files from results/*.json, filters to high-reward traces,
and converts Anthropic message format to Qwen3's multi-turn chat format
with tool-calling. Generates loss_mask (1 for assistant, 0 for user/tool).

Usage:
    python convert_traces_for_sft.py
    python convert_traces_for_sft.py --min-reward 0.7 --results-dir results
    python convert_traces_for_sft.py --results-file results/claude-opus-4-6_20260223_062030.json
"""

import argparse
import json
from pathlib import Path


# Tool definitions in OpenAI function-calling format (for chat template)
TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": "Read the Solidity code snippet under review.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Get context about the protocol being audited.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hints",
            "description": "Request detection heuristics. WARNING: reduces max reward to 0.7.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_finding",
            "description": "Submit your vulnerability analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vulnerability_type": {"type": "string"},
                    "explanation": {"type": "string"},
                    "severity": {"type": "string", "default": "HIGH"},
                    "affected_lines": {"type": "string", "default": ""},
                    "attack_path": {"type": "string", "default": ""},
                    "prerequisites": {"type": "string", "default": ""},
                    "impact": {"type": "string", "default": ""},
                },
                "required": ["vulnerability_type", "explanation"],
            },
        },
    },
]


def _build_system_prompt(scenario_info: dict) -> str:
    """Build the system prompt from trace info."""
    # We don't have full scenario data in traces, but we can reconstruct
    # a generic auditor prompt. The protocol_type isn't in the trace info
    # directly, so we use a generic version.
    return (
        "You are a smart contract security auditor.\n\n"
        "Instructions:\n"
        "1. Use read_code() to examine the Solidity code snippet\n"
        "2. Optionally use get_context() for protocol details\n"
        "3. Optionally use list_hints() for detection guidance (reduces max reward)\n"
        "4. Use submit_finding() with your analysis when ready\n\n"
        "Your goal: identify the specific vulnerability type, explain the attack vector, "
        "and pinpoint the affected lines of code."
    )


def _convert_anthropic_to_structured_messages(
    messages: list[dict],
    scenario_info: dict,
) -> list[dict]:
    """Convert Anthropic-format messages to OpenAI-structured format.

    Uses proper tool_calls fields and role=tool responses so that
    any tokenizer's apply_chat_template formats them natively
    (e.g., Qwen3.5 → qwen3_coder XML, Qwen3 → Hermes JSON).

    Anthropic input:
      - {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
      - {"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": ...}]}

    OpenAI-structured output:
      - {"role": "assistant", "content": "...", "tool_calls": [{"id": ..., "type": "function", ...}]}
      - {"role": "tool", "tool_call_id": "...", "content": "..."}
    """
    system_prompt = _build_system_prompt(scenario_info)
    out_messages = [{"role": "system", "content": system_prompt}]

    # Map tool_use_id → tool_call for linking results
    tool_use_map: dict[str, dict] = {}
    call_counter = 0

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, str):
                out_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        tool_content = block.get("content", "")
                        out_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tool_content if isinstance(tool_content, str) else json.dumps(tool_content),
                        })
                    else:
                        text = block.get("text", block.get("content", ""))
                        if text:
                            out_messages.append({"role": "user", "content": text})

        elif role == "assistant":
            if isinstance(content, str):
                out_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text.strip():
                            text_parts.append(text)
                    elif block.get("type") == "tool_use":
                        call_id = block.get("id", f"call_{call_counter}")
                        call_counter += 1
                        name = block["name"]
                        arguments = block.get("input", {})
                        tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,  # dict, not JSON string
                            },
                        })
                        tool_use_map[call_id] = tool_calls[-1]

                assistant_msg = {"role": "assistant", "content": "\n".join(text_parts)}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                out_messages.append(assistant_msg)

    return out_messages


def _is_anthropic_format(messages: list[dict]) -> bool:
    """Check if messages use Anthropic format (content is list of blocks)."""
    for m in messages:
        if isinstance(m.get("content"), list):
            return True
    return False


def _is_structured_format(messages: list[dict]) -> bool:
    """Check if messages already have structured tool_calls fields."""
    for m in messages:
        if "tool_calls" in m or m.get("role") == "tool":
            return True
    return False


def _passthrough_structured_messages(
    messages: list[dict],
    scenario_info: dict,
) -> list[dict]:
    """Pass through already-structured messages, ensuring system prompt.

    Also converts tool_calls arguments from JSON strings to dicts
    (Qwen3.5's chat template requires dict for .items() iteration).
    """
    system_prompt = _build_system_prompt(scenario_info)
    out = []
    has_system = False
    for m in messages:
        if m.get("role") == "system":
            has_system = True
            out.append({"role": "system", "content": system_prompt})
        elif "tool_calls" in m:
            # Fix arguments: JSON string → dict
            fixed_calls = []
            for tc in m["tool_calls"]:
                tc = dict(tc)
                func = dict(tc.get("function", {}))
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                func["arguments"] = args
                tc["function"] = func
                fixed_calls.append(tc)
            out.append({**m, "tool_calls": fixed_calls})
        else:
            out.append(m)
    if not has_system:
        out.insert(0, {"role": "system", "content": system_prompt})
    return out


def convert_trace(trace: dict, min_reward: float) -> dict | None:
    """Convert a single trace to structured SFT format.

    Handles both Anthropic-format (Claude) and OpenAI-format (Qwen/vLLM) traces.
    Returns None if the trace doesn't meet quality criteria.
    """
    reward = trace.get("reward", 0.0)
    if reward is None or reward < min_reward:
        return None

    messages = trace.get("messages", [])
    if not messages:
        return None

    if trace.get("error"):
        return None

    info = trace.get("info", {})
    scenario_id = trace.get("scenario_id", info.get("scenario_id", ""))

    # Auto-detect format and convert
    if _is_anthropic_format(messages):
        structured = _convert_anthropic_to_structured_messages(messages, info)
    elif _is_structured_format(messages):
        structured = _passthrough_structured_messages(messages, info)
    else:
        # Plain text messages (no tool calls) — skip
        return None

    # Validate: must have at least one tool_calls message
    has_tool_call = any("tool_calls" in m for m in structured)
    if not has_tool_call:
        return None

    return {
        "messages": structured,
        "tools": TOOLS_OPENAI,
        "reward": reward,
        "metadata": {
            "scenario_id": scenario_id,
            "source_model": trace.get("model", "unknown"),
            "canonical_category": info.get("canonical_category", ""),
            "difficulty": info.get("difficulty", "unknown"),
            "subscores": trace.get("subscores", {}),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Claude traces to SLIME SFT format")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--results-file", default=None, help="Specific results file (overrides --results-dir)")
    parser.add_argument("--output", default="data/slime_sft_traces.jsonl")
    parser.add_argument("--min-reward", type=float, default=0.7)
    args = parser.parse_args()

    # Collect result files
    if args.results_file:
        result_files = [Path(args.results_file)]
    else:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("*.json"))

    if not result_files:
        print(f"No result files found in {args.results_dir}")
        return

    # Deduplicate: keep highest-reward trace per scenario_id
    best_traces: dict[str, dict] = {}
    total_traces = 0
    total_files = 0

    for rf in result_files:
        try:
            with open(rf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {rf}: {e}")
            continue

        total_files += 1
        traces = data.get("traces", [])

        for trace in traces:
            total_traces += 1
            converted = convert_trace(trace, args.min_reward)
            if converted is None:
                continue

            sid = converted["metadata"]["scenario_id"]
            existing = best_traces.get(sid)
            if existing is None or converted["reward"] > existing["reward"]:
                best_traces[sid] = converted

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        for trace in best_traces.values():
            out.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"Processed {total_traces} traces from {total_files} files")
    print(f"Filtered to {len(best_traces)} unique scenarios with reward >= {args.min_reward}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
