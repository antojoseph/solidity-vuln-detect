"""Provider-independent audit episode helpers.

This module contains the shared prompt builder, tool schemas, tool-call parser,
and snippet-only episode state used by evaluation, rollout collection, and the
SkyRL environment. Keeping it separate avoids importing provider SDKs from core
environment code.
"""

from __future__ import annotations

import json
import re

from env import _parse_affected_lines, evaluate_submission


def build_system_prompt(protocol_type: str) -> str:
    return (
        f"You are a smart contract security auditor reviewing a {protocol_type} protocol.\n\n"
        "Instructions:\n"
        "1. Use read_code() to examine the Solidity code snippet\n"
        "2. Optionally use get_context() for protocol details\n"
        "3. Optionally use list_hints() for detection guidance (reduces max reward)\n"
        "4. Use submit_finding() with your analysis when ready\n\n"
        "Your goal: identify the specific vulnerability type, explain the attack vector, "
        "and pinpoint the affected lines of code."
    )


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
                    "description": "Comma-separated line numbers or ranges of root cause",
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


TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
    for tool in TOOL_DEFINITIONS
]


def _parse_qwen_tool_calls(content: str) -> list[dict]:
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        block = match.group(1).strip()

        func_match = re.search(r"<function=(\w+)>(.*?)</function>", block, re.DOTALL)
        if func_match:
            name = func_match.group(1)
            params_block = func_match.group(2)
            args = {}
            for param_match in re.finditer(
                r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
                params_block,
                re.DOTALL,
            ):
                args[param_match.group(1)] = param_match.group(2).strip()
            calls.append({"name": name, "arguments": args})
            continue

        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            }
        )

    return calls


class Episode:
    """Encapsulates all state for a single snippet-based audit episode."""

    def __init__(self, scenario: dict):
        self.scenario = scenario.copy()
        self.hints_used = False
        self.code_read = False
        self.submission: dict | None = None

    def read_code(self) -> str:
        self.code_read = True
        code = self.scenario.get("code_clean", "No code loaded.")
        lines = code.split("\n")
        return "\n".join(f"{i + 1:3d} | {line}" for i, line in enumerate(lines))

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
            f"  {i + 1}. {hint}" for i, hint in enumerate(hints)
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
        self.submission = {
            "vulnerability_type": vulnerability_type,
            "explanation": explanation,
            "severity": severity.upper(),
            "affected_lines": _parse_affected_lines(affected_lines),
            "attack_path": attack_path,
            "prerequisites": prerequisites,
            "impact": impact,
        }
        return f"Finding submitted: {vulnerability_type} ({severity}). Awaiting evaluation."

    def call_tool(self, name: str, args: dict) -> str:
        if name == "read_code":
            return self.read_code()
        if name == "get_context":
            return self.get_context()
        if name == "list_hints":
            return self.list_hints()
        if name == "submit_finding":
            return self.submit_finding(**args)
        return f"Unknown tool: {name}"

    def evaluate(self) -> dict:
        return evaluate_submission(
            self.scenario,
            self.submission,
            hints_used=self.hints_used,
            code_read=self.code_read,
        )