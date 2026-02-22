"""Build ood_scenarios.json from evmbench vulnerable Solidity code.

Two sources:
1. Diff reverse-application (audits with .diff files) — full contracts
2. Findings markdown extraction (audits without diffs) — code snippets

Usage:
    python build_ood_scenarios.py /path/to/evmbench/audits

Outputs:
    data/ood_scenarios.json
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Protocol-type mapping (same as build_clean_scenarios.py + non-diff audits)
# ---------------------------------------------------------------------------
AUDIT_TO_PROTOCOL_TYPE: dict[str, str] = {
    "2023-07-pooltogether": "yield",
    "2023-10-nextgen": "nft-marketplace",
    "2023-12-ethereumcreditguild": "lending",
    "2024-01-canto": "cross-chain",
    "2024-01-curves": "services",
    "2024-01-init-capital-invitational": "lending",
    "2024-01-renft": "nft-marketplace",
    "2024-02-althea-liquid-infrastructure": "liquid-staking",
    "2024-03-abracadabra-money": "lending",
    "2024-03-canto": "cross-chain",
    "2024-03-coinbase": "services",
    "2024-03-gitcoin": "governance",
    "2024-03-neobase": "dexes",
    "2024-03-taiko": "cross-chain",
    "2024-04-noya": "yield",
    "2024-05-arbitrum-foundation": "governance",
    "2024-05-loop": "yield",
    "2024-05-munchables": "gaming",
    "2024-05-olas": "staking-pool",
    "2024-06-size": "lending",
    "2024-06-thorchain": "cross-chain",
    "2024-06-vultisig": "services",
    "2024-07-basin": "dexes",
    "2024-07-benddao": "lending",
    "2024-07-munchables": "gaming",
    "2024-07-traitforge": "gaming",
    "2024-08-phi": "services",
    "2024-08-wildcat": "lending",
    "2024-12-secondswap": "dexes",
    "2025-01-liquid-ron": "liquid-staking",
    "2025-01-next-generation": "nft-marketplace",
    "2025-02-thorwallet": "cross-chain",
    "2025-04-forte": "services",
    "2025-04-virtuals": "services",
    "2025-05-blackhole": "dexes",
    "2025-06-panoptic": "derivatives",
    "2025-10-sequence": "gaming",
    "2026-01-tempo-feeamm": "dexes",
    "2026-01-tempo-mpp-streams": "services",
    "2026-01-tempo-stablecoin-dex": "dexes",
}

# ---------------------------------------------------------------------------
# Canonical category inference
# ---------------------------------------------------------------------------
# Keywords ordered so the first match wins — more specific patterns first.
CATEGORY_KEYWORD_MAP: list[tuple[str, list[str]]] = [
    ("reentrancy", ["reentrancy", "reentrant", "re-entrant", "re-entrancy", "reenter"]),
    ("flash-loan", ["flash loan", "flashloan"]),
    ("first-depositor-inflation", ["first deposit", "inflation attack"]),
    ("oracle-manipulation", ["oracle", "price manipulation", "price feed", "stale price", "twap"]),
    ("signature-replay", ["replay", "signature recovery", "bypass.*validation"]),
    ("liquidation", ["liquidat"]),
    ("erc4626-vault", ["erc4626"]),
    ("slippage-protection", ["slippage", "deadline", "amountoutmin"]),
    ("fee-on-transfer", ["fee on transfer", "deflationary token"]),
    ("denial-of-service", ["dos", "denial of service", "unbounded", "unable to", "won't be able", "cannot unstake", "incompatible", "can prevent", "griefing"]),
    ("locked-funds", ["locked", "stuck fund", "stuck in", "trapped", "irrecoverable", "lock of the", "become stuck", "loss of fund"]),
    ("integer-overflow", ["overflow", "underflow", "truncation", "uint96", "downcast"]),
    ("access-control", ["access control", "unauthorized", "permission", "anyone can", "lack of check", "missing check", "missing auth", "called by anyone", "unrestricted", "hijack", "public.*call leads"]),
    ("precision-rounding", ["precision", "rounding", "lower.*fee", "incorrect fee"]),
    ("reward-accounting", ["reward", "yield.*mismatch", "mismatch.*yield", "accounting", "claiming of fees", "claim.*fee", "slope not updated"]),
    ("stale-state", ["stale", "race condition", "outdated", "missing clear"]),
    ("initialization", ["initializ", "upgradeab"]),
    ("governance", ["governance", "voting", "proposal"]),
    ("incorrect-math", ["math", "incorrect.*calcul", "formula", "sqrt", "exp "]),
    ("unchecked-returns", ["unchecked return", "return value"]),
    ("input-validation", ["input validation", "missing validation", "invalid validation", "silently accepts"]),
]

# Manual overrides: (audit_id, vuln_id) -> canonical_category
CATEGORY_OVERRIDES: dict[tuple[str, str], str] = {
    # --- diff-based audits ---
    ("2023-07-pooltogether", "H-02"): "integer-overflow",
    ("2023-07-pooltogether", "H-04"): "access-control",
    ("2023-10-nextgen", "H-01"): "reentrancy",
    ("2023-10-nextgen", "H-02"): "access-control",
    ("2024-01-curves", "H-02"): "reward-accounting",
    ("2024-01-renft", "H-06"): "access-control",
    ("2024-03-taiko", "H-03"): "locked-funds",
    ("2024-05-olas", "H-02"): "input-validation",
    ("2024-06-size", "H-01"): "precision-rounding",
    ("2024-06-size", "H-02"): "stale-state",
    ("2024-06-size", "H-03"): "liquidation",
    ("2024-06-size", "H-04"): "liquidation",
    ("2024-07-benddao", "H-01"): "reward-accounting",
    ("2024-07-benddao", "H-02"): "access-control",
    ("2024-07-benddao", "H-03"): "stale-state",
    ("2024-07-benddao", "H-06"): "denial-of-service",
    ("2024-07-benddao", "H-07"): "access-control",
    ("2024-07-benddao", "H-08"): "denial-of-service",
    ("2024-07-basin", "H-02"): "incorrect-math",
    ("2024-08-wildcat", "H-01"): "incorrect-math",
    ("2025-01-liquid-ron", "H-01"): "erc4626-vault",
    ("2025-04-forte", "H-01"): "incorrect-math",
    ("2025-04-forte", "H-02"): "input-validation",
    ("2025-04-forte", "H-03"): "input-validation",
    ("2025-04-virtuals", "H-03"): "access-control",
    ("2025-06-panoptic", "H-01"): "incorrect-math",
    ("2025-06-panoptic", "H-02"): "incorrect-math",
    # --- markdown-based audits ---
    ("2024-01-curves", "H-03"): "access-control",
    ("2024-01-init-capital-invitational", "H-02"): "access-control",
    ("2024-01-init-capital-invitational", "H-03"): "frontrunning-mev",
    ("2024-01-renft", "H-01"): "access-control",
    ("2024-01-renft", "H-07"): "locked-funds",
    ("2024-02-althea-liquid-infrastructure", "H-01"): "reward-accounting",
    ("2024-03-abracadabra-money", "H-01"): "oracle-manipulation",
    ("2024-03-abracadabra-money", "H-03"): "locked-funds",
    ("2024-03-canto", "H-01"): "locked-funds",
    ("2024-03-gitcoin", "H-01"): "reward-accounting",
    ("2024-03-taiko", "H-01"): "incorrect-math",
    ("2024-04-noya", "H-05"): "locked-funds",
    ("2024-04-noya", "H-07"): "locked-funds",
    ("2024-04-noya", "H-09"): "locked-funds",
    ("2024-05-munchables", "H-02"): "access-control",
    ("2024-05-olas", "H-01"): "reward-accounting",
    ("2024-06-thorchain", "H-01"): "access-control",
    ("2024-06-vultisig", "H-01"): "locked-funds",
    ("2024-06-vultisig", "H-03"): "denial-of-service",
    ("2024-07-munchables", "H-01"): "access-control",
    ("2024-07-traitforge", "H-02"): "denial-of-service",
    ("2024-08-phi", "H-04"): "access-control",
    ("2024-08-phi", "H-07"): "access-control",
    ("2025-04-virtuals", "H-04"): "access-control",
    ("2025-10-sequence", "H-01"): "signature-replay",
    ("2026-01-tempo-mpp-streams", "H-03"): "signature-replay",
    ("2026-01-tempo-stablecoin-dex", "H-02"): "access-control",
    ("2026-01-tempo-stablecoin-dex", "H-03"): "incorrect-math",
    ("2026-01-tempo-stablecoin-dex", "H-04"): "denial-of-service",
}


def infer_canonical_category(
    audit_id: str, vuln_id: str, title: str
) -> str:
    """Infer a hud canonical category from the vulnerability title."""
    key = (audit_id, vuln_id)
    if key in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[key]

    title_lower = title.lower()
    for category, keywords in CATEGORY_KEYWORD_MAP:
        for kw in keywords:
            if kw in title_lower:
                return category

    return "incorrect-math"  # safe fallback for uncategorised findings


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def strip_evmbench_marker(code: str) -> str:
    """Remove the evmbench watermark comment line."""
    return "\n".join(
        line
        for line in code.split("\n")
        if not line.strip().startswith("// evmbench:")
    )


def split_diff_by_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a multi-file unified diff into per-file sections.

    Returns [(remote_path, diff_section), ...].
    """
    files: list[tuple[str, str]] = []
    current_path: str | None = None
    current_lines: list[str] = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_path and current_lines:
                files.append((current_path, "\n".join(current_lines)))
            parts = line.split(" b/")
            current_path = parts[-1] if len(parts) > 1 else None
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_path and current_lines:
        files.append((current_path, "\n".join(current_lines)))

    return files


def reverse_apply_diff(patched_content: str, diff_section: str) -> str | None:
    """Reverse-apply a single-file unified diff to patched content."""
    code_fd, code_path = tempfile.mkstemp(suffix=".sol")
    diff_fd, diff_path = tempfile.mkstemp(suffix=".diff")
    try:
        with os.fdopen(code_fd, "w") as f:
            f.write(patched_content)
        with os.fdopen(diff_fd, "w") as f:
            f.write(diff_section)

        result = subprocess.run(
            [
                "patch",
                "-R",
                "-s",
                "--no-backup-if-mismatch",
                code_path,
                diff_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        with open(code_path) as f:
            return f.read()
    except Exception:
        return None
    finally:
        for p in (code_path, diff_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def extract_bug_lines_from_diff(diff_section: str) -> list[int]:
    """Extract line numbers of vulnerable (removed) lines from a diff.

    Returns 1-indexed line numbers in the OLD (vulnerable) file.
    """
    bug_lines: list[int] = []
    old_line = 0

    for line in diff_section.split("\n"):
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)", line)
            if match:
                old_line = int(match.group(1))
        elif line.startswith("-") and not line.startswith("---"):
            bug_lines.append(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            pass  # additions don't advance old-file counter
        elif not line.startswith("\\"):
            old_line += 1

    return sorted(set(bug_lines))


# ---------------------------------------------------------------------------
# Diff-based scenario builder
# ---------------------------------------------------------------------------

def build_diff_scenarios(audit_dir: Path, config: dict) -> list[dict]:
    """Build OOD scenarios by reverse-applying diffs."""
    audit_id = config["id"]
    protocol_type = AUDIT_TO_PROTOCOL_TYPE.get(audit_id)
    if not protocol_type:
        return []

    scenarios: list[dict] = []

    for vuln in config.get("vulnerabilities", []):
        vuln_id = vuln["id"]
        title = vuln.get("title", "")
        patch_mapping = vuln.get("patch_path_mapping")
        if not patch_mapping:
            continue

        diff_path = audit_dir / "patch" / f"{vuln_id}.diff"
        if not diff_path.exists():
            continue

        diff_text = diff_path.read_text(encoding="utf-8")
        file_sections = split_diff_by_file(diff_text)

        # Build reverse mapping: remote_path -> local_patch_path
        remote_to_local: dict[str, str] = {}
        for local_p, remote_p in patch_mapping.items():
            if local_p.endswith(".sol"):
                remote_to_local[remote_p] = local_p

        # Apply reverse diff per file, pick the primary one for the scenario
        best_code: str | None = None
        all_bug_lines: list[int] = []
        best_size = 0

        for remote_path, section in file_sections:
            local_path = remote_to_local.get(remote_path)
            if not local_path:
                continue

            patched_file = audit_dir / local_path
            if not patched_file.exists():
                continue

            patched_content = strip_evmbench_marker(
                patched_file.read_text(encoding="utf-8")
            )
            vulnerable = reverse_apply_diff(patched_content, section)
            if vulnerable is None:
                continue

            section_bug_lines = extract_bug_lines_from_diff(section)
            if len(vulnerable) > best_size:
                best_code = vulnerable
                all_bug_lines = section_bug_lines
                best_size = len(vulnerable)

        if best_code is None:
            print(f"  WARN: diff reverse-apply failed for {audit_id}/{vuln_id}")
            continue

        canonical = infer_canonical_category(audit_id, vuln_id, title)
        scenario_id = f"ood/evmbench/{audit_id}/{vuln_id}"

        scenarios.append(
            {
                "id": scenario_id,
                "protocol_type": protocol_type,
                "canonical_category": canonical,
                "category_slug": canonical,
                "category_title": title,
                "code": best_code.strip(),
                "bug_lines": all_bug_lines,
                "hints": [],
                "preconditions": "",
            }
        )

    return scenarios


# ---------------------------------------------------------------------------
# Markdown-based scenario builder
# ---------------------------------------------------------------------------

_TEST_PATTERNS = re.compile(
    r"function\s+test|describe\s*\(|it\s*\(|assertEq|assert\s*\(|expect\s*\(",
    re.IGNORECASE,
)
_FIX_PATTERNS = re.compile(
    r"//\s*(FIX|FIXED|SAFE|RECOMMENDED)|FIXED:|AFTER:",
    re.IGNORECASE,
)
_SOL_INDICATORS = re.compile(
    r"pragma\s+solidity|contract\s+\w|function\s+\w|mapping\s*\(|uint256|address\s|msg\.sender|require\s*\(|emit\s+\w"
)


def extract_vulnerable_code_from_markdown(md_text: str) -> str | None:
    """Extract the first non-test, non-fix Solidity code block from a finding."""
    blocks: list[str] = []
    in_block = False
    current: list[str] = []

    for line in md_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") and not in_block:
            in_block = True
            current = []
        elif stripped.startswith("```") and in_block:
            in_block = False
            code = "\n".join(current)
            if _SOL_INDICATORS.search(code):
                blocks.append(code)
        elif in_block:
            current.append(line)

    # Filter out test and fix blocks
    candidates = [
        b
        for b in blocks
        if not _TEST_PATTERNS.search(b) and not _FIX_PATTERNS.search(b)
    ]
    if not candidates:
        # Fall back: any Solidity block that isn't a test
        candidates = [b for b in blocks if not _TEST_PATTERNS.search(b)]
    if not candidates:
        candidates = blocks

    if not candidates:
        return None

    # Return the first candidate (typically the "Vulnerability Detail" block)
    return candidates[0].strip()


def build_markdown_scenarios(audit_dir: Path, config: dict, diff_vuln_ids: set[str]) -> list[dict]:
    """Build OOD scenarios from findings markdown for vulns without diffs."""
    audit_id = config["id"]
    protocol_type = AUDIT_TO_PROTOCOL_TYPE.get(audit_id)
    if not protocol_type:
        return []

    findings_dir = audit_dir / "findings"
    if not findings_dir.exists():
        return []

    scenarios: list[dict] = []

    for vuln in config.get("vulnerabilities", []):
        vuln_id = vuln["id"]
        title = vuln.get("title", "")

        # Skip vulns already handled by diff approach
        if vuln_id in diff_vuln_ids:
            continue

        md_path = findings_dir / f"{vuln_id}.md"
        if not md_path.exists():
            continue

        md_text = md_path.read_text(encoding="utf-8")
        code = extract_vulnerable_code_from_markdown(md_text)
        if not code:
            print(f"  Skipping {audit_id}/{vuln_id}: no code block in finding")
            continue

        non_empty = [line for line in code.split("\n") if line.strip()]
        if len(non_empty) < 5:
            print(
                f"  Skipping {audit_id}/{vuln_id}: code too small ({len(non_empty)} lines)"
            )
            continue

        canonical = infer_canonical_category(audit_id, vuln_id, title)
        scenario_id = f"ood/evmbench-md/{audit_id}/{vuln_id}"

        scenarios.append(
            {
                "id": scenario_id,
                "protocol_type": protocol_type,
                "canonical_category": canonical,
                "category_slug": canonical,
                "category_title": title,
                "code": code,
                "bug_lines": [],
                "hints": [],
                "preconditions": "",
            }
        )

    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python build_ood_scenarios.py /path/to/evmbench/audits")
        sys.exit(1)

    audits_dir = Path(sys.argv[1])
    if not audits_dir.is_dir():
        print(f"Error: {audits_dir} is not a directory")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data"
    diff_scenarios: list[dict] = []
    md_scenarios: list[dict] = []

    for audit_dir in sorted(audits_dir.iterdir()):
        if not audit_dir.is_dir() or audit_dir.name == "template":
            continue

        config_path = audit_dir / "config.yaml"
        if not config_path.exists():
            continue

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        # Phase 1: diff-based scenarios
        d_scenarios = build_diff_scenarios(audit_dir, config)
        diff_vuln_ids = {
            s["id"].rsplit("/", 1)[-1] for s in d_scenarios
        }
        diff_scenarios.extend(d_scenarios)

        # Phase 2: markdown fallback for remaining vulns
        m_scenarios = build_markdown_scenarios(audit_dir, config, diff_vuln_ids)
        md_scenarios.extend(m_scenarios)

    all_scenarios = diff_scenarios + md_scenarios

    # Write output
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "ood_scenarios.json"
    with open(output_path, "w") as f:
        json.dump(all_scenarios, f, indent=2)

    # Stats
    from collections import Counter

    cat_counts = Counter(s["canonical_category"] for s in all_scenarios)
    print(f"\nDiff-based scenarios:     {len(diff_scenarios)}")
    print(f"Markdown-based scenarios: {len(md_scenarios)}")
    print(f"Total OOD scenarios:      {len(all_scenarios)}")
    print(f"\nCanonical categories ({len(cat_counts)}):")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
