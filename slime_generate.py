"""Custom generate function for SLIME RL training.

Runs the Solidity vulnerability detection agent-environment loop using
SGLang as the rollout server. Follows the retool pattern: multi-turn
tool-calling with loss_mask (1 for model tokens, 0 for tool results).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

try:
    from jinja2 import Template
except ImportError as e:
    raise ImportError("Jinja2 is required. pip install jinja2") from e

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from run_eval_standalone import Episode, TOOL_DEFINITIONS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load scenarios once
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "scenarios.json"

_scenario_map: dict[str, dict] = {}


def _get_scenario_map() -> dict[str, dict]:
    global _scenario_map
    if not _scenario_map:
        with open(DATA_PATH) as f:
            scenarios = json.load(f)
        _scenario_map = {s["id"]: s for s in scenarios}
    return _scenario_map


# ---------------------------------------------------------------------------
# Tool definitions in Qwen/OpenAI function-calling format
# ---------------------------------------------------------------------------

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "read_code",
            "description": (
                "Read the Solidity code snippet under review. "
                "Returns the smart contract code that may contain a security vulnerability."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": (
                "Get context about the protocol being audited. "
                "Returns the protocol type and preconditions. No reward penalty."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_hints",
            "description": (
                "Request detection heuristics for analyzing this code. "
                "WARNING: Using hints reduces your maximum reward from 1.0 to 0.7."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_finding",
            "description": (
                "Submit your vulnerability analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vulnerability_type": {
                        "type": "string",
                        "description": "Category of vulnerability (e.g. 'reentrancy', 'oracle manipulation')",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Detailed explanation of WHY the code is vulnerable",
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
                        "description": "Preconditions required for the exploit",
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
    },
]

# ---------------------------------------------------------------------------
# Jinja2 template for Qwen3 tool-calling format
# ---------------------------------------------------------------------------

TOOL_TEMPLATE = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

MAX_TURNS = 12


def _build_system_prompt(scenario: dict) -> str:
    protocol_type = scenario["protocol_type"]
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


def _format_prompt(system_prompt: str, tools: list[dict], messages: list[dict] | None = None) -> str:
    template = Template(TOOL_TEMPLATE)
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.append({"role": "user", "content": "Begin your audit."})
    if messages:
        msgs.extend(messages)
    return template.render(messages=msgs, tools=tools or [])


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> tags."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            raw = m.group(1).replace("\n", "\\n")
            data = json.loads(raw)
            calls.append({
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return calls


def _postprocess_response(text: str) -> str:
    """Trim response after last complete tool_call or submit_finding."""
    if "<tool_call>" in text:
        pattern = r"<tool_call>\s*\{.*?\}\s*</tool_call>"
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if matches:
            return text[:matches[-1].end()]
    return text


# ---------------------------------------------------------------------------
# Main generate function (called by SLIME)
# ---------------------------------------------------------------------------

async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Run one episode of the Solidity vuln detection environment.

    Args:
        args: SLIME rollout args (has sglang_router_ip, sglang_router_port, etc.)
        sample: Sample with prompt = scenario_id string
        sampling_params: dict of sampling parameters for SGLang

    Returns:
        Sample populated with tokens, response, loss_mask, reward, metadata
    """
    assert not args.partial_rollout, "Partial rollout not supported for this function."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # 1. Load scenario
    scenario_id = sample.prompt
    scenario_map = _get_scenario_map()
    if scenario_id not in scenario_map:
        logger.error(f"Unknown scenario_id: {scenario_id}")
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return sample

    scenario = scenario_map[scenario_id]
    ep = Episode(scenario)

    # 2. Build initial prompt
    system_prompt = _build_system_prompt(scenario)
    prompt_text = _format_prompt(system_prompt, TOOLS_OPENAI)
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # 3. Multi-turn loop
    response_text = ""
    response_token_ids: list[int] = []
    loss_masks: list[int] = []
    conversation_messages: list[dict] = []  # for building context
    tool_call_count = 0

    for turn in range(MAX_TURNS):
        # Check context length
        total_length = len(prompt_token_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_ctx = args.rollout_max_context_len
        else:
            max_ctx = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_ctx:
            sample.status = Sample.Status.TRUNCATED
            break

        # Generate from SGLang
        payload = {
            "input_ids": prompt_token_ids + response_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        output = await post(url, payload)

        # Handle abort
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        # Extract tokens and logprobs
        if "output_token_logprobs" in output["meta_info"]:
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_text = state.tokenizer.decode(cur_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            cur_text = output["text"]
            cur_text = _postprocess_response(cur_text)
            cur_token_ids = state.tokenizer(cur_text, add_special_tokens=False)["input_ids"]

        response_text += cur_text
        response_token_ids += cur_token_ids
        loss_masks += [1] * len(cur_token_ids)  # model-generated: train

        # Check length limit
        if output["meta_info"]["finish_reason"]["type"] == "length":
            break

        # Parse and execute tool calls
        tool_calls = _parse_tool_calls(cur_text)
        if not tool_calls:
            # No tool call and model stopped — episode done (no submission)
            break

        submitted = False
        tool_results = []
        for tc in tool_calls:
            result_text = ep.call_tool(tc["name"], tc["arguments"])
            tool_results.append(f'\n<tool_response>\n{{"name": "{tc["name"]}", "result": {json.dumps(result_text)}}}\n</tool_response>\n')
            tool_call_count += 1
            if tc["name"] == "submit_finding":
                submitted = True

        # Append tool results as observation (loss_mask = 0)
        obs_text = "".join(tool_results)
        # Add user turn markers for the observation
        obs_with_markers = f"<|im_end|>\n<|im_start|>user\n{obs_text}<|im_end|>\n<|im_start|>assistant\n"
        obs_token_ids = state.tokenizer(obs_with_markers, add_special_tokens=False)["input_ids"]
        response_text += obs_with_markers
        response_token_ids += obs_token_ids
        loss_masks += [0] * len(obs_token_ids)  # environment tokens: don't train

        # Pad logprobs for observation tokens
        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_token_ids)

        if submitted:
            break

    # 4. Score
    result = ep.evaluate()

    # 5. Populate sample
    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response_text
    sample.loss_mask = loss_masks
    sample.reward = result["reward"]
    sample.metadata = {
        "scenario_id": scenario_id,
        "subscores": result.get("subscores", {}),
        "info": result.get("info", {}),
        "tool_call_count": tool_call_count,
    }

    # Set status
    if sample.status == Sample.Status.PENDING:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample
