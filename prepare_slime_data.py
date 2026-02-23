"""Convert task files to SLIME prompt JSONL format.

Reads data/tasks_train.json (and optionally other task files) and writes
a JSONL file where each line has the scenario_id as the prompt text,
plus label and metadata for SLIME's Dataset class.

Usage:
    python prepare_slime_data.py
    python prepare_slime_data.py --task-file data/tasks_eval.json --output data/slime_prompts_eval.jsonl
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert task files to SLIME prompt JSONL")
    parser.add_argument("--task-file", default="data/tasks_train.json")
    parser.add_argument("--scenarios-file", default="data/scenarios.json")
    parser.add_argument("--output", default="data/slime_prompts.jsonl")
    args = parser.parse_args()

    # Load scenarios for metadata enrichment
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

            record = {
                "text": scenario_id,
                "label": scenario.get("canonical_category", ""),
                "metadata": {
                    "difficulty": scenario.get("difficulty", "unknown"),
                    "protocol_type": scenario.get("protocol_type", ""),
                    "category_slug": scenario.get("category_slug", ""),
                },
            }
            out.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} prompts to {output_path}")
    if skipped:
        print(f"Skipped {skipped} tasks (missing scenario_id or scenario not found)")


if __name__ == "__main__":
    main()
