"""SkyRL BaseTextEnv adapter for Solidity vulnerability detection.

Bridges SkyRL's Online RL training loop to the existing Episode class and
deterministic scorer from run_eval_standalone.py. This replaces SLIME's
custom generate function (slime_generate.py) with TrajGym/SkyRL's standard
BaseTextEnv interface.

Architecture:
    SkyRL SkyRLGymGenerator -> agent_loop()
        -> env.init(prompt) -> load scenario, return auditor prompt
        -> env.step(action) -> parse tool calls, dispatch to Episode
        -> env.close()      -> return final reward if not scored in step()

The env bypasses TrajGym's flag-capture reward and uses our 5-signal
deterministic scorer directly (category 40%, explanation 25%, severity 10%,
line accuracy 15%, exploitability 10%).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from run_eval_standalone import (
    Episode,
    TOOL_DEFINITIONS_OPENAI,
    build_system_prompt,
    _parse_qwen_tool_calls,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy BaseTextEnv import (same pattern as TrajGymTextEnv)
# ---------------------------------------------------------------------------

def _get_base_class():
    """Import BaseTextEnv lazily to avoid hard dep on skyrl_gym."""
    try:
        from skyrl_gym.envs.base_text_env import BaseTextEnv
        return BaseTextEnv
    except ImportError:
        logger.warning(
            "skyrl_gym not installed — SolidityVulnEnv will not register with SkyRL"
        )
        return object


_Base = _get_base_class()


def _auto_register():
    """Auto-register SolidityVulnEnv with skyrl_gym on import.

    This ensures the env is available in Ray worker processes that import
    this module via PYTHONPATH, not just the launcher process.
    """
    try:
        from skyrl_gym.envs import register as _register
        _register(
            id="solidity-vuln",
            entry_point="skyrl_env:SolidityVulnEnv",
        )
        logger.info("Auto-registered solidity-vuln env with skyrl_gym")
    except Exception:
        pass  # Will be registered explicitly by launcher if auto-register fails


_auto_register()

# ---------------------------------------------------------------------------
# Scenario data (loaded once per worker)
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "scenarios.json"

_scenario_map: dict[str, dict] = {}


def _get_scenario_map() -> dict[str, dict]:
    global _scenario_map
    if not _scenario_map:
        with open(DATA_PATH) as f:
            scenarios = json.load(f)
        _scenario_map = {s["id"]: s for s in scenarios}
        logger.info("Loaded %d scenarios from %s", len(_scenario_map), DATA_PATH)
    return _scenario_map


# ---------------------------------------------------------------------------
# Tool schemas for prompt injection
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

_TOOL_SCHEMAS_TEXT = None


def _get_tool_schemas_text() -> str:
    """Format tool schemas for injection into the system prompt."""
    global _TOOL_SCHEMAS_TEXT
    if _TOOL_SCHEMAS_TEXT is None:
        lines = [
            "\n# Tools\n",
            "You may call one or more functions to assist with the user query.\n",
            "You are provided with function signatures within <tools></tools> XML tags:",
            "<tools>",
        ]
        for tool in TOOL_DEFINITIONS_OPENAI:
            lines.append(json.dumps(tool))
        lines.append("</tools>")
        lines.append("")
        lines.append(
            "For each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:"
        )
        lines.append("<tool_call>")
        lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
        lines.append("</tool_call>")
        _TOOL_SCHEMAS_TEXT = "\n".join(lines)
    return _TOOL_SCHEMAS_TEXT


# ---------------------------------------------------------------------------
# SkyRL Environment
# ---------------------------------------------------------------------------

ConversationType = list[dict[str, Any]]

MAX_TURNS = 12


class SolidityVulnEnv(_Base):
    """SkyRL-Gym BaseTextEnv for Solidity vulnerability detection.

    Each instance manages one audit episode. Tool parsing and execution is
    handled internally (no subprocess — all tools are pure Python lookups
    against the scenario JSON). Reward is computed by the deterministic
    5-signal scorer from env.py.

    Per-sample kwargs (from dataset ``extras``):
        scenario_id: ID into scenarios.json
        difficulty: easy/medium/hard
        category: canonical vulnerability category
    """

    def __init__(
        self,
        env_config: Any = None,
        extras: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if _Base is not object:
            super().__init__()

        extras = extras or {}

        self.max_turns = int(
            extras.get("max_turns") or kwargs.get("max_turns", MAX_TURNS)
        )
        self.turns = 0

        # Per-episode data
        self._scenario_id: str = extras.get("scenario_id", "")
        self._difficulty: str | None = extras.get("difficulty")
        self._category: str | None = extras.get("category")

        # Episode state (set in init())
        self._episode: Episode | None = None
        self._done = False
        self._final_reward: float | None = None
        self._eval_result: dict | None = None

        # Track agent state for compatibility with SkyRL expectations
        self._tool_calls_history: list[dict] = []
        self._tool_outputs: list[str] = []
        self._read_code_called = False

        # Tool schemas for SkyRL (it uses these for prompt injection)
        self.tools = TOOL_DEFINITIONS_OPENAI
        self.tool_groups = []

        # Strip <think> blocks before parsing
        self._strip_think: bool = bool(
            extras.get("strip_think")
            if extras.get("strip_think") is not None
            else kwargs.get("strip_think", True)
        )

        logger.info(
            "SolidityVulnEnv created: scenario=%s max_turns=%d",
            self._scenario_id,
            self.max_turns,
        )

    def init(self, prompt: ConversationType) -> tuple:
        """Initialize episode: load scenario, return prompt with tool schemas.

        Args:
            prompt: Initial conversation (system + user messages from dataset).

        Returns:
            (prompt, metadata) — prompt with tool schemas injected.
        """
        scenario_map = _get_scenario_map()

        if self._scenario_id not in scenario_map:
            logger.error("Unknown scenario_id: %s", self._scenario_id)
            return prompt, {}

        scenario = scenario_map[self._scenario_id]
        self._episode = Episode(scenario)
        self.turns = 0
        self._done = False
        self._final_reward = None
        self._eval_result = None
        self._tool_calls_history = []
        self._tool_outputs = []

        # Build auditor system prompt with tool schemas
        system_content = build_system_prompt(scenario["protocol_type"])
        system_content += _get_tool_schemas_text()

        prompt = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Begin your audit."},
        ]

        return prompt, {"scenario_id": self._scenario_id}

    def step(self, action: str) -> dict[str, Any]:
        """Process LLM output: parse tool calls, execute, compute reward.

        Args:
            action: Raw LLM text output (may contain tool calls).

        Returns:
            dict with observations, reward, done, metadata.
        """
        self.turns += 1

        # Strip <think> blocks before parsing
        if self._strip_think:
            clean_action = _THINK_PATTERN.sub("", action).strip()
        else:
            clean_action = action

        # No episode means bad scenario_id
        if self._episode is None:
            return {
                "observations": [],
                "reward": 0.0,
                "done": True,
                "metadata": {"error": "no_episode"},
            }

        # Parse tool calls (supports both Hermes JSON and Qwen3.5 XML formats)
        tool_calls = _parse_qwen_tool_calls(clean_action)

        if not tool_calls:
            done = self.turns >= self.max_turns
            if done:
                reward = self._compute_terminal_reward()
                return {
                    "observations": [],
                    "reward": reward,
                    "done": True,
                    "metadata": self._build_metadata(),
                }
            return {
                "observations": [
                    "No tool call detected. Use read_code() to start, "
                    "then submit_finding() when ready."
                ],
                "reward": 0.0,
                "done": False,
                "metadata": {},
            }

        # Execute tool calls
        observations = []
        submitted = False

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            result_text = self._episode.call_tool(name, args)
            self._tool_calls_history.append(tc)
            self._tool_outputs.append(result_text)

            if name == "read_code":
                self._read_code_called = True

            obs = (
                f'<tool_response>\n'
                f'{{"name": "{name}", "result": {json.dumps(result_text)}}}\n'
                f'</tool_response>'
            )
            observations.append(obs)

            if name == "submit_finding":
                submitted = True

        done = submitted or self.turns >= self.max_turns

        if done:
            reward = self._compute_terminal_reward()
        else:
            reward = 0.0

        logger.info(
            "SolidityVulnEnv.step(): scenario=%s turn=%d/%d done=%s "
            "reward=%.4f tool_calls=%s submitted=%s",
            self._scenario_id,
            self.turns,
            self.max_turns,
            done,
            reward,
            [tc["name"] for tc in tool_calls],
            submitted,
        )

        return {
            "observations": observations,
            "reward": reward,
            "done": done,
            "metadata": self._build_metadata() if done else {},
        }

    def close(self):
        """Clean up episode. Compute reward if not already done."""
        if self._final_reward is None and self._episode is not None:
            self._final_reward = self._compute_terminal_reward()
            logger.info(
                "SolidityVulnEnv.close(): scenario=%s reward=%.4f (computed in close)",
                self._scenario_id,
                self._final_reward,
            )

    def _compute_terminal_reward(self) -> float:
        """Run the deterministic 5-signal scorer. Result is memoized."""
        if self._final_reward is not None:
            return self._final_reward

        if self._episode is None:
            self._final_reward = 0.0
            return 0.0

        result = self._episode.evaluate()
        reward = result["reward"]
        # Penalize blind submissions — model should read code before submitting
        if not self._read_code_called and reward > 0:
            reward *= 0.5
        self._final_reward = reward
        self._eval_result = result
        return self._final_reward

    def get_metrics(self) -> dict[str, Any]:
        """Return per-episode metrics for W&B logging."""
        metrics = {
            "turns": self.turns,
            "read_code_called": int(self._read_code_called),
            "tool_call_count": len(self._tool_calls_history),
        }
        if self._eval_result is not None:
            subscores = self._eval_result.get("subscores", {})
            for k, v in subscores.items():
                metrics[f"subscore/{k}"] = v
        return metrics

    def _build_metadata(self) -> dict:
        """Build metadata dict for the terminal step."""
        if self._eval_result is None:
            return {"scenario_id": self._scenario_id}
        return {
            "scenario_id": self._scenario_id,
            "reward": self._eval_result["reward"],
            "subscores": self._eval_result.get("subscores", {}),
            "info": self._eval_result.get("info", {}),
            "tool_call_count": len(self._tool_calls_history),
            "turns": self.turns,
        }
