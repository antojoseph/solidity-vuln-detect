"""Build clean_scenarios.json from evmbench patched Solidity files.

Usage:
    python build_clean_scenarios.py /path/to/evmbench/audits

Outputs:
    data/clean_scenarios.json
"""

import json
import re
import sys
from pathlib import Path

# Map evmbench audit IDs to protocol types matching the hud taxonomy.
AUDIT_TO_PROTOCOL_TYPE: dict[str, str] = {
    "2023-07-pooltogether": "yield",
    "2023-10-nextgen": "nft-marketplace",
    "2023-12-ethereumcreditguild": "lending",
    "2024-01-curves": "services",
    "2024-01-renft": "nft-marketplace",
    "2024-03-taiko": "cross-chain",
    "2024-04-noya": "yield",
    "2024-05-olas": "staking-pool",
    "2024-06-size": "lending",
    "2024-07-basin": "dexes",
    "2024-07-benddao": "lending",
    "2024-07-traitforge": "gaming",
    "2024-08-phi": "services",
    "2024-08-wildcat": "lending",
    "2025-01-liquid-ron": "liquid-staking",
    "2025-04-forte": "services",
    "2025-04-virtuals": "services",
    "2025-05-blackhole": "dexes",
    "2025-06-panoptic": "derivatives",
    "2026-01-tempo-feeamm": "dexes",
    "2026-01-tempo-mpp-streams": "services",
    "2026-01-tempo-stablecoin-dex": "dexes",
}

SKIP_DIRS = {"template"}
SKIP_SUFFIXES = (".hardhat.sol",)
MIN_MEANINGFUL_LINES = 10


def strip_evmbench_marker(code: str) -> str:
    """Remove the evmbench watermark comment line."""
    return "\n".join(
        line for line in code.split("\n")
        if not line.strip().startswith("// evmbench:")
    )


def is_interface_only(code: str) -> bool:
    """Return True if the file only declares interfaces (no contract/library)."""
    stripped = re.sub(r"//.*", "", code)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    stripped = re.sub(r'"[^"]*"', "", stripped)

    has_contract = bool(
        re.search(r"\b(contract|library|abstract\s+contract)\b", stripped)
    )
    has_interface = bool(re.search(r"\binterface\b", stripped))

    return has_interface and not has_contract


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python build_clean_scenarios.py /path/to/evmbench/audits")
        sys.exit(1)

    audits_dir = Path(sys.argv[1])
    if not audits_dir.is_dir():
        print(f"Error: {audits_dir} is not a directory")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data"
    scenarios: list[dict] = []

    for audit_dir in sorted(audits_dir.iterdir()):
        if not audit_dir.is_dir():
            continue
        audit_id = audit_dir.name
        if audit_id in SKIP_DIRS:
            continue

        patch_dir = audit_dir / "patch"
        if not patch_dir.exists():
            continue

        protocol_type = AUDIT_TO_PROTOCOL_TYPE.get(audit_id)
        if not protocol_type:
            print(f"WARNING: no protocol_type mapping for {audit_id}, skipping")
            continue

        for sol_file in sorted(patch_dir.glob("*.sol")):
            if sol_file.name.endswith(SKIP_SUFFIXES):
                continue

            code = sol_file.read_text(encoding="utf-8")
            code = strip_evmbench_marker(code)

            if is_interface_only(code):
                print(f"  Skipping interface-only: {audit_id}/{sol_file.name}")
                continue

            non_empty = [line for line in code.split("\n") if line.strip()]
            if len(non_empty) < MIN_MEANINGFUL_LINES:
                print(f"  Skipping too small ({len(non_empty)} lines): {audit_id}/{sol_file.name}")
                continue

            scenario_id = f"clean/evmbench/{audit_id}/{sol_file.stem}"
            scenarios.append({
                "id": scenario_id,
                "protocol_type": protocol_type,
                "code": code.strip(),
            })

    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "clean_scenarios.json"
    with open(output_path, "w") as f:
        json.dump(scenarios, f, indent=2)

    print(f"\nWrote {len(scenarios)} clean scenarios to {output_path}")


if __name__ == "__main__":
    main()
