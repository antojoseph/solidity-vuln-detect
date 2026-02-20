"""Build scenarios.json and task splits from protocol-vulnerabilities-index.

Usage:
    python build_scenarios.py /path/to/protocol-vulnerabilities-index

Outputs:
    data/scenarios.json      — All scenario entries
    data/tasks_train.json    — 80% train split (HUD task format)
    data/tasks_eval.json     — 20% eval split (HUD task format)
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical category mapping: ~305 raw slugs → ~20 classes
# ---------------------------------------------------------------------------

CANONICAL_CATEGORIES: dict[str, list[str]] = {
    "reentrancy": [
        "reentrancy", "reentrancy-via-external-calls", "reentrancy-via-token-callbacks",
        "reentrancy-in-reward-and-nft-claiming", "reentrancy-in-external-calls",
        "reentrancy-in-lending-flows", "reentrancy-and-cei-violations",
        "reentrancy-vulnerabilities", "erc777-token-hook-reentrancy",
        "reentrancy-via-callback-tokens",
    ],
    "oracle-manipulation": [
        "oracle-price-manipulation", "oracle-stale-price", "oracle-and-price-feed-issues",
        "oracle-price-feed", "oracle-price-feed-misconfiguration", "oracle-price-feed-manipulation",
        "oracle-manipulation-and-stale-prices", "oracle-price-manipulation-via-reserves",
        "oracle-stale-price-manipulation", "stale-chainlink-oracle-validation",
        "stale-oracle-data", "stale-or-manipulable-price-data", "twap-oracle-miscalculation",
        "invalid-oracle-version-handling", "incorrect-price-feed-decimal-handling",
        "oracle-and-price-manipulation",
    ],
    "access-control": [
        "access-control", "access-control-bypass", "access-control-and-privilege-escalation",
        "access-control-misconfiguration", "missing-access-control", "access-control-authorization",
        "access-bypass-restrictions", "access-control-missing", "access-control-privilege-escalation",
        "access-control-state-mismatch", "access-control-visibility-misconfiguration",
        "privilege-escalation-access-control", "privileged-role-abuse", "privileged-function-abuse",
        "irrevocable-privileged-roles", "centralization-risks", "admin-centralization-risks",
        "admin-timelock-and-recovery-abuse", "irrevocable-whitelist-or-approval",
    ],
    "flash-loan": [
        "flash-loan-attacks", "flash-loan-price-manipulation", "flash-loan-attack-vectors",
        "flash-loan-and-reward-manipulation", "flash-loan-checkpoint-manipulation",
    ],
    "first-depositor-inflation": [
        "vault-share-inflation", "first-depositor-share-inflation",
        "first-depositor-vault-share-inflation", "first-depositor-vault-share-manipulation",
        "vault-share-inflation-first-depositor", "vault-share-inflation-attack",
        "vault-share-accounting-manipulation", "share-inflation-first-depositor",
        "share-price-manipulation",
    ],
    "precision-rounding": [
        "precision-loss-rounding", "rounding-precision-loss", "rounding-and-precision-loss",
        "precision-and-rounding-errors", "decimal-handling-errors", "decimal-precision-mismatch",
        "token-decimal-mismatch", "token-decimal-assumptions", "rounding-direction-errors",
        "rounding-errors", "rounding-truncation-errors", "rounding-direction-and-precision-loss",
        "incorrect-decimal-handling", "incorrect-price-feed-decimal-handling",
        "staking-rounding-and-dust-amount-exploits", "rounding-and-precision",
    ],
    "slippage-protection": [
        "missing-slippage-protection", "slippage-and-sandwich-attacks",
        "slippage-and-price-manipulation", "slippage-protection", "slippage-protection-missing",
        "slippage-and-mev", "slippage-and-sandwich-protection", "insufficient-slippage-protection",
        "missing-slippage-deadline-protection", "frontrunning-and-slippage-protection",
    ],
    "fee-on-transfer": [
        "fee-on-transfer-incompatibility", "fee-on-transfer-token-incompatibility",
        "fee-on-transfer-token-handling", "fee-on-transfer-token-accounting",
        "fee-on-transfer-token-mishandling", "non-standard-erc20", "non-standard-token-handling",
    ],
    "integer-overflow": [
        "integer-overflow-underflow", "overflow-underflow", "unsafe-type-casting-overflow",
        "arithmetic-and-precision-errors", "arithmetic-overflow-underflow",
        "overflow-underflow-math-errors", "integer-overflow-unsafe-casting",
        "unsafe-arithmetic-in-unchecked-blocks", "unsafe-type-casting",
    ],
    "denial-of-service": [
        "denial-of-service", "denial-of-service-griefing", "denial-of-service-gas-griefing",
        "denial-of-service-via-unbounded-operations", "denial-of-service-unbounded-loops",
        "withdrawal-queue-dos", "griefing-attacks", "dos-gas-griefing",
        "unbounded-loop-dos", "unbounded-loop-dos-in-reward-claims",
        "unbounded-loops-denial-of-service", "unbounded-loops-gas-exhaustion",
        "gas-griefing-eip150", "gas-limit-and-estimation", "liquidation-auction-dos",
        "gas-griefing-and-denial-of-service",
    ],
    "frontrunning-mev": [
        "front-running-and-mev", "front-running", "frontrunning-mev",
        "front-running-sandwich-attacks", "front-running-sandwich",
        "frontrunning-attacks", "frontrunning-initialization",
        "frontrunning-state-invalidation", "frontrunning-unprotected-state-transitions",
    ],
    "governance": [
        "governance-voting-manipulation", "governance-vote-manipulation",
        "governance-voting-flaws", "governance-voting", "governance-voting-checkpoint",
        "governance-voting-power-manipulation", "voting-power-manipulation",
    ],
    "liquidation": [
        "liquidation-mechanism-flaws", "liquidation-logic-flaws", "liquidation-logic",
        "liquidation-logic-errors", "liquidation-process-failures",
        "liquidation-vulnerabilities",
    ],
    "input-validation": [
        "missing-input-validation", "insufficient-input-validation", "input-validation",
        "input-validation-missing-checks", "missing-input-validation-in-admin-setters",
        "missing-existence-validation-on-state-operations",
    ],
    "reward-accounting": [
        "reward-accounting-errors", "reward-distribution-errors", "reward-distribution-flaws",
        "reward-distribution-staking-flaws", "reward-distribution", "reward-distribution-issues",
        "reward-token-accounting", "reward-accounting-manipulation",
        "staking-reward-calculation-errors", "incorrect-fee-reward-accounting",
        "incorrect-fee-reward-calculation",
    ],
    "unchecked-returns": [
        "unchecked-return-values", "unchecked-external-calls", "unsafe-external-calls",
        "unsafe-erc20-operations", "unsafe-erc20-token-handling", "unsafe-erc20-handling",
        "unsafe-erc20-token-transfers", "unsafe-erc20-approval", "unsafe-token-approvals",
        "unsafe-erc721-operations", "unsafe-nft-minting", "unsafe-nft-minting-transfer",
        "unsafe-eth-transfer-methods", "unchecked-external-call-return-values",
    ],
    "initialization": [
        "initialization-and-upgrade-flaws", "initialization-vulnerabilities",
        "initialization-and-upgradeability", "initialization-upgrade-vulnerabilities",
        "initialization-proxy-vulnerabilities", "initialization-upgrade",
        "initialization-and-upgrade-vulnerabilities", "upgradeable-storage-gap",
    ],
    "erc4626-vault": [
        "erc4626-vault-compliance", "erc4626-vault-share-issues", "erc4626-vault-edge-cases",
        "erc4626-vault-integration", "erc4626-vault",
    ],
    "locked-funds": [
        "locked-funds", "funds-locked-in-contracts", "locked-or-frozen-funds",
        "fund-locking-trapped-assets", "locked-stuck-funds", "locked-and-irretrievable-funds",
        "locked-or-stuck-funds", "funds-permanently-locked", "stuck-or-locked-funds",
        "locked-or-lost-funds", "fund-lock-griefing", "fund-lock",
        "fund-lock-and-stuck-tokens", "funds-locked-in-edge-case-states",
        "excess-msg-value-not-refunded", "excess-msg-value-locked",
        "eth-handling-and-overpayment-loss", "eth-handling-refund-issues",
        "eth-handling-and-refund-errors", "native-eth-handling",
    ],
    "stale-state": [
        "missing-state-updates", "stale-state-missing-updates", "stale-state-dependency",
        "stale-protocol-state", "state-update-inconsistency", "state-update-ordering",
        "state-update-ordering-errors", "state-update-after-external-call",
        "incorrect-state-updates", "incorrect-state-accounting", "stale-state-after-operations",
        "stale-cached-state-desync", "stale-state-after-actions",
        "state-inconsistency-after-lifecycle-transitions",
        "state-inconsistency-on-withdrawal",
    ],
    "signature-replay": [
        "signature-and-replay-vulnerabilities", "signature-replay-attacks",
        "signature-replay-and-validation", "signature-replay-and-theft",
        "signature-hash-verification", "signature-replay",
        "replay-attack-missing-nonce", "replay-and-signature-vulnerabilities",
        "cross-chain-message-replay", "cross-chain-message-verification",
    ],
    "incorrect-math": [
        "incorrect-math-calculations", "incorrect-math-and-accounting",
        "incorrect-arithmetic-logic", "incorrect-fee-reward-accounting",
        "incorrect-collateral-valuation",
    ],
}

# Build reverse lookup
_SLUG_TO_CANONICAL: dict[str, str] = {}
for canonical, slugs in CANONICAL_CATEGORIES.items():
    for slug in slugs:
        _SLUG_TO_CANONICAL[slug] = canonical


def strip_annotations(code: str) -> str:
    """Remove VULNERABLE and BUG annotations from code."""
    lines = []
    for line in code.split("\n"):
        # Skip lines that are just VULNERABLE comments
        if re.match(r"^\s*//\s*VULNERABLE:", line):
            continue
        # Remove inline BUG comments
        line = re.sub(r"\s*//\s*BUG:.*$", "", line)
        lines.append(line)
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_bug_lines(code_raw: str) -> list[int]:
    """Find line numbers (1-indexed) with // BUG: annotations."""
    bug_lines = []
    for i, line in enumerate(code_raw.split("\n"), 1):
        if "// BUG:" in line:
            bug_lines.append(i)
    return bug_lines


def assign_difficulty(code_clean: str, bug_line_count: int) -> str:
    """Assign difficulty based on code length and annotation density."""
    lines = [l for l in code_clean.split("\n") if l.strip()]
    n = len(lines)
    if n <= 10:
        return "easy"
    elif n <= 25:
        return "medium"
    else:
        return "hard"


def parse_category_file(filepath: Path, protocol_type: str) -> list[dict]:
    """Parse a single category markdown file into scenario entries."""
    content = filepath.read_text(encoding="utf-8")
    category_slug = filepath.stem

    # Extract title
    title_match = re.match(r"# (.+)", content)
    category_title = title_match.group(1).strip() if title_match else category_slug

    # Extract Preconditions section
    precond_match = re.search(
        r"## Preconditions\n(.*?)(?=\n## )", content, re.DOTALL
    )
    preconditions = precond_match.group(1).strip() if precond_match else ""

    # Extract Vulnerable Pattern section
    vuln_match = re.search(
        r"## Vulnerable Pattern\n(.*?)(?=\n## )", content, re.DOTALL
    )
    if not vuln_match:
        return []
    vuln_section = vuln_match.group(1)

    # Extract Detection Heuristics
    hints_match = re.search(
        r"## Detection Heuristics\n(.*?)(?=\n## )", content, re.DOTALL
    )
    hints = []
    if hints_match:
        for line in hints_match.group(1).strip().split("\n"):
            line = re.sub(r"^\d+\.\s*", "", line.strip())
            if line:
                hints.append(line)

    # Extract all solidity code blocks from the Vulnerable Pattern section
    code_blocks = re.findall(r"```solidity\n(.*?)```", vuln_section, re.DOTALL)
    if not code_blocks:
        return []

    # Map to canonical category
    canonical = _SLUG_TO_CANONICAL.get(category_slug, category_slug)

    scenarios = []
    for i, code_raw in enumerate(code_blocks):
        code_raw = code_raw.strip()
        code_clean = strip_annotations(code_raw)

        # Skip if stripping removed all meaningful code
        if len(code_clean.strip()) < 20:
            continue

        bug_lines = extract_bug_lines(code_raw)
        difficulty = assign_difficulty(code_clean, len(bug_lines))

        scenarios.append({
            "id": f"{protocol_type}/{category_slug}/{i}",
            "protocol_type": protocol_type,
            "category_slug": category_slug,
            "category_title": category_title,
            "canonical_category": canonical,
            "code_clean": code_clean,
            "code_raw": code_raw,
            "hints": hints,
            "preconditions": preconditions,
            "bug_lines": bug_lines,
            "difficulty": difficulty,
        })

    return scenarios


def build_task_entry(scenario_id: str) -> dict:
    """Build a HUD-compatible task entry."""
    return {
        "scenario": "detect-vuln",
        "args": {"scenario_id": scenario_id},
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_scenarios.py /path/to/protocol-vulnerabilities-index")
        sys.exit(1)

    repo_path = Path(sys.argv[1])
    categories_dir = repo_path / "categories"
    if not categories_dir.exists():
        print(f"Error: {categories_dir} not found")
        sys.exit(1)

    # Parse all category files
    all_scenarios = []
    for protocol_dir in sorted(categories_dir.iterdir()):
        if not protocol_dir.is_dir():
            continue
        protocol_type = protocol_dir.name
        for md_file in sorted(protocol_dir.glob("*.md")):
            scenarios = parse_category_file(md_file, protocol_type)
            all_scenarios.extend(scenarios)

    print(f"Parsed {len(all_scenarios)} scenarios from {categories_dir}")

    # Stats
    canonical_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for s in all_scenarios:
        canonical_counts[s["canonical_category"]] += 1
        difficulty_counts[s["difficulty"]] += 1

    print(f"\nCanonical categories ({len(canonical_counts)}):")
    for cat, count in sorted(canonical_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    # Stratified train/eval split (80/20)
    random.seed(42)
    by_category = defaultdict(list)
    for s in all_scenarios:
        by_category[s["canonical_category"]].append(s)

    train_ids = []
    eval_ids = []
    for cat, items in by_category.items():
        random.shuffle(items)
        split_point = max(1, int(len(items) * 0.8))
        for s in items[:split_point]:
            train_ids.append(s["id"])
        for s in items[split_point:]:
            eval_ids.append(s["id"])

    print(f"\nSplit: {len(train_ids)} train, {len(eval_ids)} eval")

    # Write outputs
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "scenarios.json", "w") as f:
        json.dump(all_scenarios, f, indent=2)
    print(f"Wrote {out_dir / 'scenarios.json'}")

    train_tasks = [build_task_entry(sid) for sid in train_ids]
    with open(out_dir / "tasks_train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"Wrote {out_dir / 'tasks_train.json'} ({len(train_tasks)} tasks)")

    eval_tasks = [build_task_entry(sid) for sid in eval_ids]
    with open(out_dir / "tasks_eval.json", "w") as f:
        json.dump(eval_tasks, f, indent=2)
    print(f"Wrote {out_dir / 'tasks_eval.json'} ({len(eval_tasks)} tasks)")


if __name__ == "__main__":
    main()
