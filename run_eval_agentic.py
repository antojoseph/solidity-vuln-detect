"""Agentic security audit eval harness.

Evaluates an LLM's ability to discover Solidity vulnerabilities by giving it
access to full project repos with rich exploration tools (list files, read files,
grep). Inspired by the Forge Proof agentic harness.

Scores against the same deterministic ground truth as run_eval_standalone.py.

Examples:
    # Qwen at local endpoint
    python run_eval_agentic.py --base-url http://linux:30000/v1 --model Qwen/Qwen3.5-397B-A17B-FP8 --max-tasks 5

    # Full eval
    python run_eval_agentic.py --base-url http://linux:30000/v1 --model Qwen/Qwen3.5-397B-A17B-FP8

    # Include snippet-only scenarios (no repo, fallback to basic tools)
    python run_eval_agentic.py --base-url http://linux:30000/v1 --model Qwen/Qwen3.5-397B-A17B-FP8 --include-snippets
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Mock the hud package so env.py can be imported without hud-python installed.
# We only use the pure scoring functions from env.py, not the HUD framework.
if "hud" not in sys.modules:
    _hud = types.ModuleType("hud")
    _hud.Environment = type("Environment", (), {
        "__init__": lambda self, *a, **k: None,
        "tool": lambda self: (lambda f: f),
        "scenario": lambda self, *a: (lambda f: f),
        "run": lambda self, **k: None,
    })
    sys.modules["hud"] = _hud
    _hud_tools = types.ModuleType("hud.tools")
    sys.modules["hud.tools"] = _hud_tools
    _hud_tools_types = types.ModuleType("hud.tools.types")
    _hud_tools_types.EvaluationResult = type("EvaluationResult", (), {})
    _hud_tools_types.SubScore = type("SubScore", (), {})
    sys.modules["hud.tools.types"] = _hud_tools_types

from dotenv import load_dotenv
from openai import AsyncOpenAI

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
# Auto-submit extraction from text analysis
# ---------------------------------------------------------------------------

# Category keywords for extraction
_VULN_TYPES = [
    "reentrancy", "oracle-manipulation", "access-control", "flash-loan",
    "precision-rounding", "slippage-protection", "fee-on-transfer",
    "integer-overflow", "denial-of-service", "frontrunning-mev",
    "governance", "liquidation", "input-validation", "reward-accounting",
    "unchecked-returns", "initialization", "erc4626-vault", "locked-funds",
    "stale-state", "signature-replay", "incorrect-math", "no-vulnerability",
    "first-depositor-inflation",
]


def _auto_submit_from_text(ep, text: str) -> None:
    """Best-effort extraction of a vulnerability finding from free text.

    Called when the model wrote analysis but never called submit_finding().
    Extracts what it can and submits — score will be partial but non-zero.
    """
    text_lower = text.lower()

    # Find vulnerability type
    vuln_type = "input-validation"  # fallback
    for vt in _VULN_TYPES:
        if vt.replace("-", " ") in text_lower or vt in text_lower:
            vuln_type = vt
            break

    # Find severity
    severity = "HIGH"
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        if sev.lower() in text_lower:
            severity = sev
            break

    # Find line numbers (patterns like "line 42", "lines 10-15", "L91")
    import re
    line_nums = re.findall(r'(?:line|L|#)\s*(\d+)', text, re.IGNORECASE)
    affected_lines = ",".join(line_nums[:10]) if line_nums else ""

    # Use the text itself as explanation (truncate if needed)
    explanation = text[:2000] if len(text) > 2000 else text

    ep.submit_finding(
        vulnerability_type=vuln_type,
        explanation=explanation,
        severity=severity,
        affected_lines=affected_lines,
        attack_path="",
        prerequisites="",
        impact="",
    )


# ---------------------------------------------------------------------------
# Qwen native tool-call parsing (from run_eval_standalone.py)
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

        # Try Qwen3.5 XML format
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

        # Try JSON format
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
# Tool definitions (OpenAI function calling format)
# ---------------------------------------------------------------------------

AGENTIC_TOOL_DEFINITIONS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List Solidity (.sol) files in the project. "
                "Returns file paths with line counts. "
                "Optionally specify a subdirectory to narrow the listing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Subdirectory to list (default: project root)",
                        "default": "",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the project with line numbers. "
                "Use this to examine Solidity contract source code. "
                "Returns the file content with numbered lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file (e.g. 'contracts/Vault.sol')",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": (
                "Search for a regex pattern across Solidity files in the project. "
                "Returns matching lines with file paths and line numbers. "
                "Useful for tracing function calls, state variable usage, "
                "imports, and vulnerability patterns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (case-insensitive)",
                    },
                    "filepath": {
                        "type": "string",
                        "description": "Optional: search only this specific file",
                        "default": "",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_info",
            "description": (
                "Get an overview of the project: protocol type, name, project structure, "
                "and which file to focus your audit on."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
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
                "(e.g. '5,6,7'). Use the line numbers from read_file().\n"
                "    attack_path: Step-by-step outline of how the exploit is executed.\n"
                "    prerequisites: Preconditions required for the exploit to work.\n"
                "    impact: Expected impact if exploited (e.g. fund loss, stolen assets)."
            ),
            "parameters": {
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
    },
]

# Fallback tool definitions for snippet-only scenarios (no repo)
SNIPPET_TOOL_DEFINITIONS_OPENAI = [
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
                "Returns the protocol type and preconditions."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    AGENTIC_TOOL_DEFINITIONS_OPENAI[-1],  # submit_finding (same)
]

# Exclude dirs for file operations
_EXCLUDE_DIRS = {"node_modules", ".git", "test", "tests", "mock", "mocks", "lib", "out", "cache"}


# ---------------------------------------------------------------------------
# Agentic episode (full repo access)
# ---------------------------------------------------------------------------


class AgenticEpisode:
    """Episode with full project repo access and rich tools."""

    def __init__(self, scenario: dict, repo_root: Path):
        self.scenario = scenario.copy()
        self.repo_root = repo_root.resolve()
        self.source_file = scenario.get("source_file", "")
        self.submission: dict | None = None
        self.tool_call_count = 0
        self.files_read: set[str] = set()

    def _safe_path(self, filepath: str) -> Path | None:
        """Resolve a path and verify it's within repo_root."""
        try:
            target = (self.repo_root / filepath).resolve()
        except (ValueError, OSError):
            return None
        if not str(target).startswith(str(self.repo_root)):
            return None
        return target

    def list_files(self, directory: str = "") -> str:
        self.tool_call_count += 1
        target = self._safe_path(directory)
        if target is None:
            return "Error: invalid directory path"
        if not target.is_dir():
            return f"Error: not a directory: {directory}"

        sol_files = []
        for f in sorted(target.rglob("*.sol")):
            rel_parts = f.relative_to(self.repo_root).parts
            if any(part in _EXCLUDE_DIRS for part in rel_parts):
                continue
            rel = str(f.relative_to(self.repo_root))
            try:
                lines = len(f.read_text(errors="replace").split("\n"))
            except Exception:
                lines = 0
            sol_files.append(f"{rel}  ({lines} lines)")

        if not sol_files:
            return "No .sol files found in this directory."
        output = f"Found {len(sol_files)} Solidity files:\n"
        output += "\n".join(sol_files[:200])
        if len(sol_files) > 200:
            output += f"\n... and {len(sol_files) - 200} more"
        return output

    def read_file(self, filepath: str) -> str:
        self.tool_call_count += 1
        target = self._safe_path(filepath)
        if target is None:
            return "Error: invalid file path"
        if not target.exists():
            return f"Error: file not found: {filepath}"
        if not target.is_file():
            return f"Error: not a file: {filepath}"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        self.files_read.add(filepath)
        lines = content.split("\n")
        max_lines = 1000
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[:max_lines]))
        if len(lines) > max_lines:
            numbered += f"\n\n... truncated ({len(lines)} total lines, showing first {max_lines})"
        return numbered

    def grep_code(self, pattern: str, filepath: str = "") -> str:
        self.tool_call_count += 1
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = []
        if filepath:
            target = self._safe_path(filepath)
            files_to_search = [target] if target and target.is_file() else []
        else:
            files_to_search = sorted(self.repo_root.rglob("*.sol"))

        for f in files_to_search:
            rel_parts = f.relative_to(self.repo_root).parts
            if any(part in _EXCLUDE_DIRS for part in rel_parts):
                continue
            try:
                content = f.read_text(errors="replace")
            except Exception:
                continue
            rel = str(f.relative_to(self.repo_root))
            for i, line in enumerate(content.split("\n"), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{i}  {line.strip()}")
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break

        if not matches:
            return f"No matches found for pattern: {pattern}"
        output = f"Found {len(matches)} match{'es' if len(matches) != 1 else ''}:\n"
        output += "\n".join(matches)
        if len(matches) >= 50:
            output += "\n... (results capped at 50)"
        return output

    def get_project_info(self) -> str:
        self.tool_call_count += 1
        protocol_type = self.scenario.get("protocol_type", "unknown")
        protocol_name = self.scenario.get("protocol_name", "")

        sol_count = sum(
            1 for f in self.repo_root.rglob("*.sol")
            if not any(p in _EXCLUDE_DIRS for p in f.relative_to(self.repo_root).parts)
        )

        top_level = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in self.repo_root.iterdir()
            if p.name not in (".git", "node_modules", ".github")
        )

        info = f"Protocol type: {protocol_type}\n"
        if protocol_name:
            info += f"Protocol name: {protocol_name}\n"
        info += f"\nProject root contents:\n"
        info += "\n".join(f"  {name}" for name in top_level[:30])
        info += f"\n\nTotal Solidity files: {sol_count}"
        info += f"\nFocus your audit on: {self.source_file}"
        return info

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
        self.tool_call_count += 1
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
        if name == "list_files":
            return self.list_files(args.get("directory", ""))
        elif name == "read_file":
            return self.read_file(args.get("filepath", ""))
        elif name == "grep_code":
            return self.grep_code(args.get("pattern", ""), args.get("filepath", ""))
        elif name == "get_project_info":
            return self.get_project_info()
        elif name == "submit_finding":
            return self.submit_finding(**args)
        else:
            return f"Unknown tool: {name}"

    def evaluate(self) -> dict:
        """Deterministic scoring — identical to Episode.evaluate() in run_eval_standalone.py."""
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
        final = round(raw * schema_penalty, 4)

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
                "hints_used": False,
                "code_read": True,
                "schema_penalty": schema_penalty,
                "submitted_type": self.submission["vulnerability_type"],
                "submitted_severity": self.submission["severity"],
                "submitted_lines": self.submission["affected_lines"],
                "ground_truth_lines": self.scenario.get("bug_lines", []),
            },
        }


# ---------------------------------------------------------------------------
# Snippet-only episode (fallback for scenarios without repos)
# ---------------------------------------------------------------------------


class SnippetEpisode:
    """Minimal episode for snippet-only scenarios (no repo)."""

    def __init__(self, scenario: dict):
        self.scenario = scenario.copy()
        self.submission: dict | None = None
        self.code_read = False

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

    def submit_finding(self, **kwargs) -> str:
        parsed_lines = []
        affected = kwargs.get("affected_lines", "")
        if isinstance(affected, str) and affected.strip():
            for part in affected.split(","):
                part = part.strip()
                if part.isdigit():
                    parsed_lines.append(int(part))

        self.submission = {
            "vulnerability_type": kwargs.get("vulnerability_type", ""),
            "explanation": kwargs.get("explanation", ""),
            "severity": kwargs.get("severity", "HIGH").upper(),
            "affected_lines": parsed_lines,
            "attack_path": kwargs.get("attack_path", ""),
            "prerequisites": kwargs.get("prerequisites", ""),
            "impact": kwargs.get("impact", ""),
        }
        return f"Finding submitted: {self.submission['vulnerability_type']} ({self.submission['severity']}). Awaiting evaluation."

    def call_tool(self, name: str, args: dict) -> str:
        if name == "read_code":
            return self.read_code()
        elif name == "get_context":
            return self.get_context()
        elif name == "submit_finding":
            return self.submit_finding(**args)
        return f"Unknown tool: {name}"

    def evaluate(self) -> dict:
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
        final = round(raw * schema_penalty, 4)

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
                "hints_used": False,
                "code_read": self.code_read,
                "schema_penalty": schema_penalty,
                "submitted_type": self.submission["vulnerability_type"],
                "submitted_severity": self.submission["severity"],
                "submitted_lines": self.submission["affected_lines"],
                "ground_truth_lines": self.scenario.get("bug_lines", []),
            },
        }


# ---------------------------------------------------------------------------
# Agent loop — agentic episode (full repo)
# ---------------------------------------------------------------------------


async def run_agentic_episode(
    client: AsyncOpenAI,
    model: str,
    scenario: dict,
    repo_root: Path,
    max_steps: int = 20,
) -> dict:
    """Run a single agentic eval episode via OpenAI-compatible API."""
    ep = AgenticEpisode(scenario, repo_root)
    protocol_type = scenario["protocol_type"]
    source_file = scenario.get("source_file", "")

    system_prompt = (
        f"You are an expert smart contract security auditor. You have access to a full "
        f"Solidity project ({protocol_type} protocol).\n\n"
        f"Your task: Identify the security vulnerability in this project, focusing on "
        f"the file '{source_file}'.\n\n"
        f"## Approach\n"
        f"1. Start by getting project info and listing files to understand the architecture\n"
        f"2. Read the focus file carefully\n"
        f"3. Trace external calls, state variables, and access control patterns\n"
        f"4. Read related contracts that interact with the focus file\n"
        f"5. Search for specific patterns (reentrancy, unchecked returns, oracle issues, etc.)\n"
        f"6. Submit your finding with a specific vulnerability type, detailed explanation, "
        f"severity, affected line numbers, attack path, prerequisites, and impact\n\n"
        f"## Analysis Checklist\n"
        f"- State variables: who can modify them? Read/write ordering?\n"
        f"- External calls: before or after state changes? Callback potential?\n"
        f"- Access control: are modifiers applied consistently?\n"
        f"- Math: division before multiplication? Unchecked blocks?\n"
        f"- Token handling: fee-on-transfer? Rebasing? ERC777 hooks?\n"
        f"- Oracle usage: stale prices? Manipulation vectors?\n\n"
        f"## Important\n"
        f"- Line numbers in your finding should reference '{source_file}'\n"
        f"- Be specific about the vulnerability type (e.g., 'reentrancy', 'oracle-manipulation', "
        f"'access-control', not generic 'security issue')\n"
        f"- Explain the attack vector step by step\n"
        f"- You MUST call submit_finding() when you are ready"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Begin your security audit."},
    ]
    steps = 0
    error = None

    try:
        for step in range(max_steps):
            steps = step + 1

            response = await client.chat.completions.create(
                model=model,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                messages=messages,
                tools=AGENTIC_TOOL_DEFINITIONS_OPENAI,
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
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    }
                    # Add step urgency for late steps
                    if steps >= max_steps - 3 and tc.function.name != "submit_finding":
                        tool_msg["content"] += (
                            f"\n\n[Step {steps}/{max_steps}] Running low on steps. "
                            f"Submit your finding soon using submit_finding()."
                        )
                    messages.append(tool_msg)
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
                # Model output text without a tool call. Instead of giving up,
                # nudge it to submit — the text often contains a valid analysis.
                if step < max_steps - 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[Step {steps}/{max_steps}] You wrote an analysis but "
                            f"didn't call a tool. Please call submit_finding() now "
                            f"with your vulnerability assessment. Include: "
                            f"vulnerability_type, explanation, severity, affected_lines, "
                            f"attack_path, prerequisites, and impact."
                        ),
                    })
                    continue  # Give it another chance
                break  # Final steps exhausted, give up

            submitted = False
            tool_results = []
            for tc in tool_calls:
                result_text = ep.call_tool(tc["name"], tc["arguments"])
                tool_results.append(
                    f'{{"name": "{tc["name"]}", "content": {json.dumps(result_text)}}}'
                )
                if tc["name"] == "submit_finding":
                    submitted = True

            # Add step counter to tool responses for urgency
            step_prefix = f"[Step {steps}/{max_steps}] "
            results_text = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>" for r in tool_results
            )
            if steps >= max_steps - 3 and not submitted:
                results_text += (
                    f"\n\n{step_prefix}You are running low on steps. "
                    f"Submit your finding soon using submit_finding()."
                )
            messages.append({"role": "user", "content": results_text})

            if submitted:
                break

    except Exception as e:
        error = str(e)

    # Safety net: if the model never submitted but wrote analysis text,
    # try to extract a submission from the last assistant message.
    if ep.submission is None and messages:
        last_assistant = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content", "").strip():
                last_assistant = m["content"]
                break
        if last_assistant and len(last_assistant) > 50:
            _auto_submit_from_text(ep, last_assistant)

    result = ep.evaluate()
    result["messages"] = messages
    result["steps"] = steps
    result["error"] = error
    result["model"] = model
    result["scenario_id"] = scenario["id"]
    result["repo_key"] = str(repo_root.relative_to(repo_root.parent.parent))
    result["files_read"] = sorted(ep.files_read)
    result["tool_call_count"] = ep.tool_call_count
    result["auto_submitted"] = ep.submission is not None and result.get("reward", 0) > 0

    return result


# ---------------------------------------------------------------------------
# Agent loop — snippet episode (fallback)
# ---------------------------------------------------------------------------


async def run_snippet_episode(
    client: AsyncOpenAI,
    model: str,
    scenario: dict,
    max_steps: int = 10,
) -> dict:
    """Run a snippet-only episode (no repo) via OpenAI-compatible API."""
    ep = SnippetEpisode(scenario)
    protocol_type = scenario["protocol_type"]

    system_prompt = (
        f"You are a smart contract security auditor reviewing a {protocol_type} protocol.\n\n"
        f"Instructions:\n"
        f"1. Use read_code() to examine the Solidity code snippet\n"
        f"2. Optionally use get_context() for protocol details\n"
        f"3. Use submit_finding() with your analysis when ready\n\n"
        f"Your goal: identify the specific vulnerability type, explain the attack vector, "
        f"and pinpoint the affected lines of code."
    )

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
                max_tokens=4096,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                messages=messages,
                tools=SNIPPET_TOOL_DEFINITIONS_OPENAI,
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""

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

            tool_calls = _parse_qwen_tool_calls(content)
            messages.append({"role": "assistant", "content": content})

            if not tool_calls:
                break

            submitted = False
            tool_results = []
            for tc in tool_calls:
                result_text = ep.call_tool(tc["name"], tc["arguments"])
                tool_results.append(
                    f'{{"name": "{tc["name"]}", "content": {json.dumps(result_text)}}}'
                )
                if tc["name"] == "submit_finding":
                    submitted = True

            results_text = "\n".join(
                f"<tool_response>\n{r}\n</tool_response>" for r in tool_results
            )
            messages.append({"role": "user", "content": results_text})

            if submitted:
                break

    except Exception as e:
        error = str(e)

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
    concurrency: int = 4,
    max_steps: int = 20,
    base_url: str = "http://linux:30000/v1",
    resume_from: str | None = None,
    include_snippets: bool = False,
    repo_mapping_path: str = "data/repo_mapping.json",
):
    """Run agentic evaluation across models and tasks."""
    api_key = os.environ.get("OPENAI_API_KEY", "empty")
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    print(f"Provider: OpenAI-compatible ({base_url})")

    # Load repo mapping
    mapping_path = Path(repo_mapping_path)
    if not mapping_path.exists():
        print(f"Error: {mapping_path} not found. Run: python build_repo_mapping.py")
        sys.exit(1)
    with open(mapping_path) as f:
        repo_mapping = json.load(f)
    print(f"Repo mapping: {len(repo_mapping)} entries")

    repos_dir = Path("data/repos")

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

    # Resolve tasks into (scenario, repo_path|None) pairs
    agentic_tasks = []
    snippet_tasks = []
    skipped = 0
    for t in tasks:
        sid = t.get("args", {}).get("scenario_id", "")
        scenario = scenario_map.get(sid)
        if not scenario:
            skipped += 1
            continue

        repo_key = repo_mapping.get(sid)
        if repo_key:
            repo_path = repos_dir / repo_key
            if repo_path.exists():
                agentic_tasks.append((scenario, repo_path))
                continue

        # No repo — snippet fallback
        if include_snippets:
            snippet_tasks.append(scenario)
        else:
            skipped += 1

    all_task_list = [(s, rp, True) for s, rp in agentic_tasks] + [(s, None, False) for s in snippet_tasks]
    print(f"Agentic tasks (full repo): {len(agentic_tasks)}")
    if include_snippets:
        print(f"Snippet tasks (fallback): {len(snippet_tasks)}")
    print(f"Skipped (no repo/scenario): {skipped}")
    print(f"Total episodes: {len(all_task_list) * len(models)}")
    print(f"Concurrency: {concurrency}")
    print()

    semaphore = asyncio.Semaphore(concurrency)

    for model in models:
        cached_results = []
        pending_tasks = []
        for scenario, repo_path, is_agentic in all_task_list:
            if scenario["id"] in prev_results:
                cached_results.append(prev_results[scenario["id"]])
            else:
                pending_tasks.append((scenario, repo_path, is_agentic))

        total = len(pending_tasks)
        completed = 0

        print(f"=== {model} ===")
        if cached_results:
            print(f"  Reusing {len(cached_results)} results from previous run")
        print(f"  Running {total} remaining episodes\n")

        async def run_with_semaphore(
            model: str,
            scenario: dict,
            repo_path: Path | None,
            is_agentic: bool,
        ) -> dict:
            nonlocal completed
            async with semaphore:
                if is_agentic and repo_path:
                    result = await run_agentic_episode(
                        client, model, scenario, repo_path, max_steps=max_steps
                    )
                else:
                    result = await run_snippet_episode(
                        client, model, scenario, max_steps=min(max_steps, 10)
                    )
                completed += 1
                reward = result["reward"]
                sid = result["scenario_id"]
                mode = "A" if is_agentic else "S"
                status = f"E {result['error'][:40]}" if result["error"] else f"R={reward:.3f}"
                print(f"  [{completed}/{total}] {model} | [{mode}] {sid} | {status}")
                return result

        coros = [
            run_with_semaphore(model, s, rp, ia)
            for s, rp, ia in pending_tasks
        ]
        new_results = await asyncio.gather(*coros)

        results = cached_results + list(new_results)

        # Aggregate
        rewards = [r["reward"] for r in results if r.get("error") is None]
        errors = [r for r in results if r.get("error") is not None]

        by_category = defaultdict(list)
        by_difficulty = defaultdict(list)
        for r in results:
            if r.get("error") is None:
                cat = r.get("info", {}).get("canonical_category", "unknown")
                diff = r.get("info", {}).get("difficulty", "unknown")
                by_category[cat].append(r["reward"])
                by_difficulty[diff].append(r["reward"])

        # Agentic stats
        agentic_results = [r for r in results if r.get("repo_key") and r.get("error") is None]
        agentic_stats = {}
        if agentic_results:
            agentic_stats = {
                "avg_steps": round(statistics.mean(r["steps"] for r in agentic_results), 1),
                "avg_files_read": round(
                    statistics.mean(len(r.get("files_read", [])) for r in agentic_results), 1
                ),
                "avg_tool_calls": round(
                    statistics.mean(r.get("tool_call_count", 0) for r in agentic_results), 1
                ),
            }

        summary = {
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_file": task_file,
            "eval_type": "agentic",
            "total_tasks": len(all_task_list),
            "agentic_tasks": len(agentic_tasks),
            "snippet_tasks": len(snippet_tasks),
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
            "agentic_stats": agentic_stats,
            "traces": results,
        }

        # Write results
        safe_model = model.replace("/", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = Path("results") / f"agentic_{safe_model}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n--- {model} Summary ---")
        print(f"  Completed: {len(rewards)}/{len(all_task_list)}")
        print(f"  Errors: {len(errors)}")
        if rewards:
            print(f"  Mean reward: {summary['mean_reward']:.4f} ± {summary['stdev_reward']:.4f}")
            print(f"  Median reward: {summary['median_reward']:.4f}")
        if agentic_stats:
            print(f"  Avg steps: {agentic_stats['avg_steps']}")
            print(f"  Avg files read: {agentic_stats['avg_files_read']}")
            print(f"  Avg tool calls: {agentic_stats['avg_tool_calls']}")
        print(f"  Results saved to: {out_path}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Agentic security audit eval harness")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B-FP8")
    parser.add_argument("--base-url", default="http://linux:30000/v1",
                        help="OpenAI-compatible base URL")
    parser.add_argument("--task-file", default="data/tasks_eval.json")
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Max agent turns per episode (default: 20)")
    parser.add_argument("--resume", default=None,
                        help="Resume from a previous results file")
    parser.add_argument("--repo-mapping", default="data/repo_mapping.json",
                        help="Path to scenario->repo mapping file")
    parser.add_argument("--include-snippets", action="store_true",
                        help="Also run snippet-only scenarios with basic tools")
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
        include_snippets=args.include_snippets,
        repo_mapping_path=args.repo_mapping,
    ))


if __name__ == "__main__":
    main()
