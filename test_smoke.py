"""Smoke test for the Solidity vuln detection training pipeline.

Verifies:
1. SolidityVulnEnv loads, creates episodes, scores correctly
2. Data files are present and well-formed
3. BaseTextEnv interface contract (init/step/close)
4. Reward signal has variance across different submissions

Run: python3 test_smoke.py
"""

import json
import sys
from pathlib import Path

# Track pass/fail
passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}{f': {detail}' if detail else ''}")


def main():
    global passed, failed

    print("=" * 60)
    print("Solidity Vuln Detection — Pipeline Smoke Test")
    print("=" * 60)

    # --- 1. Imports ---
    print("\n--- Imports ---")
    try:
        from skyrl_env import SolidityVulnEnv
        check("import SolidityVulnEnv", True)
    except Exception as e:
        check("import SolidityVulnEnv", False, str(e))
        print("\nCritical import failure. Aborting.")
        sys.exit(1)

    try:
        from skyrl_gym.envs.base_text_env import BaseTextEnv
        check("SolidityVulnEnv inherits BaseTextEnv",
              issubclass(SolidityVulnEnv, BaseTextEnv))
    except ImportError:
        check("skyrl_gym available", False, "skyrl-gym not installed")

    try:
        from run_eval_standalone import Episode, build_system_prompt, TOOL_DEFINITIONS_OPENAI
        check("import shared utilities", True)
    except Exception as e:
        check("import shared utilities", False, str(e))

    # --- 2. Data files ---
    print("\n--- Data Files ---")
    data_dir = Path("data")

    scenarios_path = data_dir / "scenarios.json"
    check("data/scenarios.json exists", scenarios_path.exists())

    if scenarios_path.exists():
        with open(scenarios_path) as f:
            scenarios = json.load(f)
        check(f"scenarios loaded ({len(scenarios)} total)", len(scenarios) > 0)

        # Check schema
        s = scenarios[0]
        required_keys = ["id", "protocol_type", "canonical_category", "code_clean"]
        missing = [k for k in required_keys if k not in s]
        check("scenario schema valid", len(missing) == 0,
              f"missing keys: {missing}")

    skyrl_path = data_dir / "skyrl_prompts.jsonl"
    if skyrl_path.exists():
        with open(skyrl_path) as f:
            lines = f.readlines()
        check(f"skyrl_prompts.jsonl ({len(lines)} lines)", len(lines) > 0)

        sample = json.loads(lines[0])
        check("skyrl prompt has 'prompt' key", "prompt" in sample)
        check("skyrl prompt has 'extras' key", "extras" in sample)
        check("extras has scenario_id",
              "scenario_id" in sample.get("extras", {}))
    else:
        check("skyrl_prompts.jsonl exists", False,
              "run: python3 prepare_skyrl_data.py")

    sft_path = data_dir / "slime_sft_traces.jsonl"
    if sft_path.exists():
        with open(sft_path) as f:
            first = json.loads(f.readline())
        check(f"slime_sft_traces.jsonl exists", True)
        check("SFT trace has messages", "messages" in first)
    else:
        check("slime_sft_traces.jsonl exists", False, "optional — needed for SFT")

    # --- 3. Episode lifecycle ---
    print("\n--- Episode Lifecycle ---")

    if not scenarios_path.exists():
        print("  SKIP  (no scenarios.json)")
    else:
        sid = scenarios[0]["id"]
        cat = scenarios[0]["canonical_category"]

        env = SolidityVulnEnv(extras={"scenario_id": sid, "max_turns": 10})
        check("env created", env is not None)

        prompt, meta = env.init([])
        check("init() returns 2 messages", len(prompt) == 2)
        check("init() system message has content",
              len(prompt[0]["content"]) > 100)
        check("init() meta has scenario_id",
              meta.get("scenario_id") == sid)

        # Step: read_code
        r1 = env.step(
            '<tool_call>\n{"name": "read_code", "arguments": {}}\n</tool_call>'
        )
        check("read_code: not done", not r1["done"])
        check("read_code: has observations", len(r1["observations"]) > 0)
        check("read_code: reward=0", r1["reward"] == 0.0)

        # Step: submit correct category
        r2 = env.step(
            f'<tool_call>\n{{"name": "submit_finding", "arguments": '
            f'{{"vulnerability_type": "{cat}", '
            f'"explanation": "The contract has a {cat} vulnerability where external calls are made before state updates, allowing attackers to drain funds.", '
            f'"severity": "HIGH", '
            f'"affected_lines": "5,6,7", '
            f'"attack_path": "Deploy malicious contract, call vulnerable function, reenter", '
            f'"prerequisites": "Contract must hold funds", '
            f'"impact": "Complete fund drainage"}}}}\n</tool_call>'
        )
        check("submit: done=True", r2["done"])
        check("submit: reward > 0", r2["reward"] > 0)
        check("submit: has subscores",
              "subscores" in r2.get("metadata", {}))

        if "subscores" in r2.get("metadata", {}):
            ss = r2["metadata"]["subscores"]
            check("category_match > 0 (correct category)",
                  ss.get("category_match", 0) > 0)

        env.close()
        check("close() succeeds", True)

    # --- 4. Reward variance ---
    print("\n--- Reward Variance ---")

    if not scenarios_path.exists():
        print("  SKIP  (no scenarios.json)")
    else:
        rewards = []
        test_scenarios = scenarios[:5]

        for s in test_scenarios:
            env = SolidityVulnEnv(
                extras={"scenario_id": s["id"], "max_turns": 5}
            )
            env.init([])
            env.step(
                '<tool_call>\n{"name": "read_code", "arguments": {}}\n</tool_call>'
            )
            # Submit wrong category to test partial scoring
            r = env.step(
                '<tool_call>\n{"name": "submit_finding", "arguments": '
                '{"vulnerability_type": "reentrancy", '
                '"explanation": "Found reentrancy in external call", '
                '"severity": "HIGH"}}\n</tool_call>'
            )
            rewards.append(r["reward"])
            env.close()

        check(f"got rewards for {len(rewards)} episodes",
              len(rewards) == len(test_scenarios))

        unique_rewards = len(set(rewards))
        check(f"reward variance ({unique_rewards} unique values from {len(rewards)} episodes)",
              unique_rewards > 1,
              f"all rewards identical: {rewards[0]:.4f}" if unique_rewards == 1 else "")

        min_r = min(rewards)
        max_r = max(rewards)
        check(f"reward range: [{min_r:.4f}, {max_r:.4f}]",
              max_r <= 1.0 and min_r >= 0.0)

    # --- 5. System prompt ---
    print("\n--- System Prompt ---")
    from run_eval_standalone import build_system_prompt
    prompt = build_system_prompt("defi")
    check("build_system_prompt works", "security auditor" in prompt)
    check("prompt mentions protocol type", "defi" in prompt)

    # --- Summary ---
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All checks passed!")
    else:
        print(f"WARNING: {failed} check(s) failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
