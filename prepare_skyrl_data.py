"""Convert task files to SkyRL Online RL JSONL format.

Reads data/tasks_train.json and writes a JSONL file where each line has
the prompt messages (system + user) and extras with scenario metadata.
This replaces prepare_slime_data.py for the SkyRL training pipeline.

Output format (per line):
{
  "prompt": [
    {"role": "system", "content": "You are a smart contract security auditor..."},
    {"role": "user", "content": "Begin your audit."}
  ],
  "extras": {
    "scenario_id": "findings/algo-stables/6354/0",
    "difficulty": "easy",
    "category": "oracle-manipulation",
    "ground_truth_flag": ""
  }
}

Note: ground_truth_flag is empty because we use a custom continuous reward
function (5-signal deterministic scorer) instead of TrajGym's flag-capture
reward. SkyRL requires the field to exist but we bypass it in SolidityVulnEnv.

Usage:
    python prepare_skyrl_data.py
    python prepare_skyrl_data.py --task-file data/tasks_eval.json --output data/skyrl_eval.jsonl
"""

import argparse
import json
from pathlib import Path

from run_eval_standalone import build_system_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Convert task files to SkyRL Online RL JSONL"
    )
    parser.add_argument("--task-file", default="data/tasks_train.json")
    parser.add_argument("--scenarios-file", default="data/scenarios.json")
    parser.add_argument("--output", default="data/skyrl_prompts.jsonl")
    args = parser.parse_args()

    # Load scenarios for metadata and prompt building
    with open(args.scenarios_file) as f:
        scenarios = json.load(f)
    scenario_map = {s["id"]: s for s in scenarios}

    # Load tasks
    with open(args.task_file) as f:
        tasks = json.load(f)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(output_path, "w") as out:
        for task in tasks:
            scenario_id = task.get("args", {}).get("scenario_id", "")
            if not scenario_id:
                skipped += 1
                continue

            scenario = scenario_map.get(scenario_id)
            if scenario is None:
                skipped += 1
                continue

            system_prompt = build_system_prompt(scenario["protocol_type"])

            record = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Begin your audit."},
                ],
                "extras": {
                    "scenario_id": scenario_id,
                    "difficulty": scenario.get("difficulty", "unknown"),
                    "category": scenario.get("canonical_category", ""),
                    "ground_truth_flag": "",  # Not used; custom reward
                },
            }
            out.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {output_path}")
    if skipped:
        print(f"Skipped {skipped} tasks (missing scenario_id or scenario not found)")


if __name__ == "__main__":
    main()
