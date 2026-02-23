"""Build scenarios.json and task splits from real audit findings.

Reads data/findings/protocols/*.json from the protocol-vulnerabilities-index repo
and loads actual .sol source files from cloned contest repos (see fetch_repos.py).

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
# GitHub URL parsing
# ---------------------------------------------------------------------------

GITHUB_SOL_URL_RE = re.compile(
    r"https://github\.com/"
    r"(?P<org>[^/]+)/(?P<repo>[^/]+)"
    r"/blob/(?P<ref>[^/]+)/"
    r"(?P<filepath>[^\s)>\]#]+\.sol)"
    r"(?:#L(?P<start>\d+)(?:-L(?P<end>\d+))?)?"
)

# ---------------------------------------------------------------------------
# Canonical category mapping (reused from original)
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

_SLUG_TO_CANONICAL: dict[str, str] = {}
for _canonical, _slugs in CANONICAL_CATEGORIES.items():
    for _slug in _slugs:
        _SLUG_TO_CANONICAL[_slug] = _canonical

# ---------------------------------------------------------------------------
# Tag → canonical category mapping
# ---------------------------------------------------------------------------

TAG_TO_CANONICAL: dict[str, str] = {
    # Reentrancy
    "Reentrancy": "reentrancy",
    "Read-only Reentrancy": "reentrancy",
    # Oracle
    "Oracle": "oracle-manipulation",
    "Stale Price": "oracle-manipulation",
    "Chainlink": "oracle-manipulation",
    # Access control
    "Access Control": "access-control",
    "Admin": "access-control",
    "Can't Remove Access Control": "access-control",
    "Update State After Admin Action": "access-control",
    # Flash loan
    "Flash Loan": "flash-loan",
    # First depositor
    "First Depositor Issue": "first-depositor-inflation",
    "Share Inflation": "first-depositor-inflation",
    # Precision / rounding
    "Decimals": "precision-rounding",
    "Rounding": "precision-rounding",
    "Precision Loss": "precision-rounding",
    "Time Rounding": "precision-rounding",
    # Slippage
    "Slippage": "slippage-protection",
    "Sandwich Attack": "slippage-protection",
    # Fee on transfer
    "Fee On Transfer": "fee-on-transfer",
    # Overflow
    "Overflow/Underflow": "integer-overflow",
    # DoS
    "DOS": "denial-of-service",
    "Denial-Of-Service": "denial-of-service",
    "Grief Attack": "denial-of-service",
    # Frontrunning
    "Front-Running": "frontrunning-mev",
    # Governance
    "Vote": "governance",
    # Liquidation
    "Liquidation": "liquidation",
    # Input validation
    "Validation": "input-validation",
    "Min/Max Cap Validation": "input-validation",
    "Change Validation": "input-validation",
    "Data Validation": "input-validation",
    "Missing Check": "input-validation",
    "MinOut/MaxIn Validation": "input-validation",
    # Reward accounting
    "Deposit/Reward tokens": "reward-accounting",
    # Unchecked returns
    "Check Return Value": "unchecked-returns",
    # Initialization
    "Initialization": "initialization",
    "Initializer": "initialization",
    "onlyInitializing modifier": "initialization",
    "initializer modifier": "initialization",
    # ERC4626
    "ERC4626": "erc4626-vault",
    # Locked funds
    "Fund Lock": "locked-funds",
    # Stale state
    "Don't update state": "stale-state",
    # Signature replay
    "Replay Attack": "signature-replay",
    "Signature Malleability": "signature-replay",
    # Incorrect math
    "Wrong Math": "incorrect-math",
}

# Keywords for fallback category matching (from env.py)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "reentrancy": [
        "reentrancy", "reentrant", "re-enter", "callback",
        "external call before state", "checks-effects-interactions", "CEI",
    ],
    "oracle-manipulation": [
        "oracle", "price feed", "stale price", "chainlink", "twap",
        "latestAnswer", "latestRoundData", "price manipulation",
    ],
    "access-control": [
        "access control", "authorization", "privilege", "onlyOwner",
        "modifier", "permissioned", "admin", "unauthorized",
    ],
    "flash-loan": [
        "flash loan", "flashloan", "single transaction", "atomic",
        "borrow and repay", "flash mint",
    ],
    "first-depositor-inflation": [
        "first depositor", "share inflation", "vault inflation",
        "donation attack", "dead shares", "empty vault",
    ],
    "precision-rounding": [
        "precision", "rounding", "truncation", "decimal",
        "division before multiplication", "loss of precision",
    ],
    "slippage-protection": [
        "slippage", "minimum output", "amountOutMin", "sandwich",
        "frontrun", "deadline", "max slippage",
    ],
    "fee-on-transfer": [
        "fee on transfer", "deflationary", "rebasing",
        "non-standard erc20", "actual amount", "transfer amount mismatch",
    ],
    "integer-overflow": [
        "overflow", "underflow", "unchecked", "type casting",
        "unsafe cast", "truncation",
    ],
    "denial-of-service": [
        "denial of service", "dos", "gas limit", "unbounded loop",
        "griefing", "block gas", "out of gas",
    ],
    "frontrunning-mev": [
        "frontrun", "front-run", "MEV", "sandwich", "mempool",
        "transaction ordering", "block builder",
    ],
    "governance": [
        "governance", "voting", "proposal", "quorum", "delegate",
        "vote manipulation", "flash vote",
    ],
    "liquidation": [
        "liquidation", "liquidate", "collateral", "health factor",
        "underwater", "bad debt", "insolvency",
    ],
    "input-validation": [
        "input validation", "missing check", "require", "assert",
        "boundary", "zero address", "zero amount",
    ],
    "reward-accounting": [
        "reward", "staking reward", "distribution", "accrual",
        "reward per share", "earned", "claim",
    ],
    "unchecked-returns": [
        "unchecked return", "return value", "safeTransfer",
        "approve", "low-level call", "success check",
    ],
    "initialization": [
        "initialize", "initializer", "proxy", "upgrade",
        "implementation", "uninitialized", "reinitialize",
    ],
    "erc4626-vault": [
        "erc4626", "vault", "share price", "convertToShares",
        "convertToAssets", "maxDeposit", "maxWithdraw",
    ],
    "locked-funds": [
        "locked", "stuck", "trapped", "irrecoverable",
        "cannot withdraw", "permanently lost", "excess ETH",
    ],
    "stale-state": [
        "stale", "outdated", "not updated", "missing update",
        "cached", "desync", "inconsistent state",
    ],
    "signature-replay": [
        "replay", "signature", "nonce", "ecrecover",
        "EIP-712", "permit", "cross-chain replay",
    ],
    "incorrect-math": [
        "incorrect calculation", "wrong formula", "math error",
        "off by one", "accounting error",
    ],
}

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

# Patterns indicating JS/TS test code (not Solidity)
_JS_INDICATORS = re.compile(
    r"(?:describe\s*\(|it\s*\(|it\.only\s*\(|await\s|ethers\.|expect\s*\(|"
    r"const\s+\{|require\s*\(\s*['\"]|import\s+\{|async\s+function|"
    r"\.connect\s*\(|\.deploy\s*\(|hardhat|waffle|chai|mocha)",
    re.IGNORECASE,
)

# Patterns indicating Solidity code
_SOL_INDICATORS = re.compile(
    r"(?:function\s+\w+|contract\s+\w+|pragma\s+solidity|mapping\s*\(|"
    r"uint256\s|address\s|bytes32\s|modifier\s+\w+|event\s+\w+|"
    r"require\s*\([^'\"]*,|emit\s+\w+|storage\s|memory\s|calldata\s)",
)

# Audit markers in code
_AUDIT_MARKER_RE = re.compile(
    r"//\s*(?:<={2,}|<--\s*|@audit\b|@audit-issue\b|@audit-info\b|BUG:|VULNERABLE:)",
    re.IGNORECASE,
)

# Code block extraction from markdown
_CODE_BLOCK_RE = re.compile(
    r"```(?:solidity|sol|)?\s*\n(.*?)```",
    re.DOTALL,
)

# Section headers in finding content
_MITIGATION_SECTION_RE = re.compile(
    r"###?\s*(?:Recommended\s+Mitigation|Mitigation|Fix(?:ed)?|Remediation)\b",
    re.IGNORECASE,
)

_POC_SECTION_RE = re.compile(
    r"###?\s*(?:Proof\s+of\s+Concept|PoC|Test)\b",
    re.IGNORECASE,
)


def _is_solidity(code: str) -> bool:
    """Check if code block looks like Solidity (not JS/TS test code)."""
    if _JS_INDICATORS.search(code):
        return False
    return bool(_SOL_INDICATORS.search(code))


def _extract_audit_markers(code: str) -> list[int]:
    """Find line numbers (1-indexed) with audit markers."""
    lines = []
    for i, line in enumerate(code.split("\n"), 1):
        if _AUDIT_MARKER_RE.search(line):
            lines.append(i)
    return lines


def _strip_audit_markers(code: str) -> str:
    """Remove audit marker comments from code."""
    lines = []
    for line in code.split("\n"):
        # Skip lines that are ONLY an audit marker comment
        stripped = line.strip()
        if stripped.startswith("//") and _AUDIT_MARKER_RE.search(stripped):
            # Check if it's a standalone comment line (no code before it)
            if re.match(r"^\s*//", line):
                continue
        # Remove inline audit markers
        line = re.sub(r"\s*//\s*(?:<={2,}|<--\s*).*$", "", line)
        line = re.sub(r"\s*//\s*@audit\b.*$", "", line)
        line = re.sub(r"\s*//\s*@audit-issue\b.*$", "", line)
        line = re.sub(r"\s*//\s*@audit-info\b.*$", "", line)
        line = re.sub(r"\s*//\s*BUG:.*$", "", line)
        lines.append(line)
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_solidity_from_content(content: str) -> list[tuple[str, list[int]]]:
    """Extract Solidity code blocks from finding content.

    Returns list of (code, audit_marker_lines) tuples.
    Filters out JS/TS test code and mitigation/fix code.
    """
    # Find where mitigation section starts (we want code BEFORE it)
    mitigation_start = len(content)
    m = _MITIGATION_SECTION_RE.search(content)
    if m:
        mitigation_start = m.start()

    # Also note PoC section (deprioritize code from there)
    poc_start = len(content)
    m = _POC_SECTION_RE.search(content)
    if m:
        poc_start = m.start()

    results = []
    for m in _CODE_BLOCK_RE.finditer(content):
        code = m.group(1).strip()
        block_pos = m.start()

        # Skip code in mitigation section
        if block_pos >= mitigation_start:
            continue

        # Skip non-Solidity code
        if not _is_solidity(code):
            continue

        # Skip very short blocks
        non_empty = [l for l in code.split("\n") if l.strip()]
        if len(non_empty) < 5:
            continue

        # Deprioritize PoC section code (but still include if it's Solidity)
        is_poc = block_pos >= poc_start

        audit_lines = _extract_audit_markers(code)
        results.append((code, audit_lines, is_poc))

    # Sort: non-PoC first, then by length (longer = more complete)
    results.sort(key=lambda x: (x[2], -len(x[0])))

    # Return without the is_poc flag
    return [(code, lines) for code, lines, _ in results]


def parse_source_refs(content: str) -> list[dict]:
    """Parse GitHub source code URLs from finding content.

    Returns list of {org, repo, filepath, start_line, end_line} dicts.
    Only includes refs that appear BEFORE the mitigation section.
    """
    mitigation_start = len(content)
    m = _MITIGATION_SECTION_RE.search(content)
    if m:
        mitigation_start = m.start()

    refs = []
    seen = set()
    for m in GITHUB_SOL_URL_RE.finditer(content):
        if m.start() >= mitigation_start:
            continue
        key = (m.group("org"), m.group("repo"), m.group("filepath"))
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "org": m.group("org"),
            "repo": m.group("repo"),
            "filepath": m.group("filepath"),
            "start_line": int(m.group("start")) if m.group("start") else None,
            "end_line": int(m.group("end")) if m.group("end") else None,
        })
    return refs


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

def map_to_canonical(tags: list[str], title: str, summary: str) -> tuple[str, str]:
    """Map finding to canonical category.

    Returns (canonical_category, category_slug).
    """
    # Layer 1: tag lookup
    for tag in tags:
        canonical = TAG_TO_CANONICAL.get(tag)
        if canonical:
            slug = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
            return canonical, slug

    # Layer 2: keyword matching on title + summary
    text = f"{title} {summary}".lower()
    best_cat = None
    best_hits = 0
    for canonical, keywords in CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text)
        if hits > best_hits:
            best_hits = hits
            best_cat = canonical

    if best_cat and best_hits >= 2:
        return best_cat, best_cat

    # Single keyword hit — lower confidence but still use it
    if best_cat and best_hits == 1:
        return best_cat, best_cat

    # Fallback: slugify the first tag or use "unknown"
    if tags:
        slug = re.sub(r"[^a-z0-9]+", "-", tags[0].lower()).strip("-")
        return slug, slug

    return "unknown", "unknown"


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------

def assign_difficulty(code: str) -> str:
    """Assign difficulty based on code length (fallback only).

    This is used when no eval data is available. When eval results exist,
    use `relabel_difficulty()` to assign empirical difficulty based on
    actual model performance.
    """
    lines = [l for l in code.split("\n") if l.strip()]
    n = len(lines)
    if n <= 10:
        return "easy"
    elif n <= 25:
        return "medium"
    else:
        return "hard"


# Per-category difficulty derived from Opus 4.6 + Sonnet 4.6 eval results.
# Based on average reward tertiles across 965 evaluated scenarios:
#   easy  >= 0.641 avg reward (both models find the vuln reliably)
#   medium  [0.478, 0.641)   (partial detection)
#   hard  <  0.478           (models consistently struggle)
CATEGORY_DIFFICULTY: dict[str, str] = {
    # easy — models score well
    "denial-of-service": "easy",
    "oracle-manipulation": "easy",
    "access-control": "easy",
    "precision-rounding": "easy",
    "signature-replay": "easy",
    "frontrunning-mev": "easy",
    "fee-on-transfer": "easy",
    # medium — partial detection
    "slippage-protection": "medium",
    "first-depositor-inflation": "medium",
    "flash-loan": "medium",
    "reentrancy": "medium",
    "stale-state": "medium",
    "integer-overflow": "medium",
    "reward-accounting": "medium",
    "input-validation": "medium",
    "locked-funds": "medium",
    "governance": "medium",
    "incorrect-math": "medium",
    "initialization": "medium",
    "liquidation": "medium",
    "unchecked-returns": "medium",
    "erc4626-vault": "medium",
    # hard — models consistently struggle
    "no-vulnerability": "hard",
    "business-logic": "hard",
}


def relabel_difficulty(scenarios: list[dict], results_dir: str = "results") -> int:
    """Relabel scenario difficulty using eval results.

    For scenarios with direct eval data, uses average reward across models.
    For unevaluated scenarios, falls back to CATEGORY_DIFFICULTY mapping,
    then to code-length heuristic.

    Returns number of scenarios relabeled.
    """
    import glob
    import statistics

    # Load all result files and build per-scenario reward map
    reward_map: dict[str, list[float]] = {}
    for path in glob.glob(f"{results_dir}/*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            # Skip small test runs
            if data.get("total_tasks", 0) < 100:
                continue
            for trace in data.get("traces", []):
                if trace.get("error") is None:
                    sid = trace["scenario_id"]
                    reward_map.setdefault(sid, []).append(trace["reward"])
        except (json.JSONDecodeError, KeyError):
            continue

    if not reward_map:
        return 0

    # Compute per-scenario average reward
    avg_rewards = {sid: statistics.mean(rs) for sid, rs in reward_map.items()}

    # Tertile thresholds from evaluated scenarios
    sorted_rewards = sorted(avg_rewards.values())
    n = len(sorted_rewards)
    t_hard = sorted_rewards[n // 3]
    t_easy = sorted_rewards[2 * n // 3]

    changed = 0
    for s in scenarios:
        avg = avg_rewards.get(s["id"])
        if avg is not None:
            # Direct eval data
            new_diff = "hard" if avg < t_hard else ("easy" if avg >= t_easy else "medium")
        elif s["canonical_category"] in CATEGORY_DIFFICULTY:
            # Category-level fallback
            new_diff = CATEGORY_DIFFICULTY[s["canonical_category"]]
        else:
            # Code-length fallback for unknown categories
            new_diff = assign_difficulty(s.get("code_clean", ""))

        if s.get("difficulty") != new_diff:
            s["difficulty"] = new_diff
            changed += 1

    return changed


# ---------------------------------------------------------------------------
# Optional scenario loaders (clean / OOD)
# ---------------------------------------------------------------------------

def _load_optional_scenarios(path: Path, scenario_type: str) -> list[dict]:
    """Load clean_scenarios.json or ood_scenarios.json if present."""
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = []
    for entry in data:
        scenario_id = entry.get("id")
        protocol_type = entry.get("protocol_type")
        code = entry.get("code")
        if not scenario_id or not protocol_type or not code:
            print(f"Skipping malformed {scenario_type} entry: {entry.get('id', '?')}")
            continue

        if scenario_type == "clean":
            canonical = "no-vulnerability"
            category_slug = "no-vulnerability"
            category_title = "No vulnerability"
            bug_lines: list[int] = []
        else:
            canonical = entry.get("canonical_category")
            if not canonical:
                print(f"Skipping OOD entry missing canonical_category: {entry.get('id')}")
                continue
            category_slug = entry.get("category_slug", canonical)
            category_title = entry.get("category_title", canonical)
            bug_lines = entry.get("bug_lines", [])

        code_clean = _strip_audit_markers(code)
        difficulty = assign_difficulty(code_clean)

        scenarios.append({
            "id": scenario_id,
            "protocol_type": protocol_type,
            "category_slug": category_slug,
            "category_title": category_title,
            "canonical_category": canonical,
            "code_clean": code_clean,
            "code_raw": code,
            "hints": entry.get("hints", []),
            "preconditions": entry.get("preconditions", ""),
            "bug_lines": bug_lines,
            "difficulty": difficulty,
        })

    return scenarios


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_task_entry(scenario_id: str) -> dict:
    """Build a HUD-compatible task entry."""
    return {
        "scenario": "detect-vuln",
        "args": {"scenario_id": scenario_id},
    }


def process_finding(
    finding: dict,
    protocol_type: str,
    repos_dir: Path,
) -> list[dict]:
    """Process a single finding into scenario entries.

    Returns 0 or more scenario dicts.
    """
    finding_id = finding.get("id", "")
    content = finding.get("content", "")
    title = finding.get("title", "")
    summary = finding.get("summary", "")
    tags = finding.get("tags", [])
    impact = finding.get("impact", "").upper()
    quality = finding.get("quality_score", 0)
    protocol_name = finding.get("protocol_name", "")
    firm_name = finding.get("firm_name", "")

    # Quality filter
    if quality < 3:
        return []

    # Map category
    canonical, category_slug = map_to_canonical(tags, title, summary)

    scenarios = []

    # Strategy 1: Try to load full .sol files from cloned repos
    source_refs = parse_source_refs(content)
    repo_scenarios_created = False

    for ref_idx, ref in enumerate(source_refs[:2]):  # Max 2 files per finding
        repo_path = repos_dir / ref["org"] / ref["repo"]
        file_path = repo_path / ref["filepath"]

        if not file_path.exists():
            continue

        try:
            code_raw = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Skip files that are too large (>1000 lines) or too small (<5 lines)
        line_count = len(code_raw.split("\n"))
        if line_count > 1000 or line_count < 5:
            continue

        # Derive bug_lines from the URL line range
        bug_lines = []
        if ref["start_line"]:
            start = ref["start_line"]
            end = ref["end_line"] or start
            bug_lines = list(range(start, end + 1))

        # Also check for audit markers in the code itself
        audit_markers = _extract_audit_markers(code_raw)
        if audit_markers and not bug_lines:
            bug_lines = audit_markers

        code_clean = _strip_audit_markers(code_raw)
        difficulty = assign_difficulty(code_clean)

        scenario_id = f"findings/{protocol_type}/{finding_id}/{ref_idx}"
        scenarios.append({
            "id": scenario_id,
            "protocol_type": protocol_type,
            "category_slug": category_slug,
            "category_title": title,
            "canonical_category": canonical,
            "code_clean": code_clean,
            "code_raw": code_raw,
            "hints": [],
            "preconditions": summary,
            "bug_lines": bug_lines,
            "difficulty": difficulty,
            "ground_truth_severity": impact,
            "finding_id": finding_id,
            "protocol_name": protocol_name,
            "firm_name": firm_name,
            "quality_score": quality,
            "source_file": ref["filepath"],
        })
        repo_scenarios_created = True

    # Strategy 2: Fallback to extracting code from content
    if not repo_scenarios_created:
        code_blocks = extract_solidity_from_content(content)
        for block_idx, (code_raw, audit_lines) in enumerate(code_blocks[:1]):  # Max 1 from content
            code_clean = _strip_audit_markers(code_raw)

            # Skip very short extracts
            non_empty = [l for l in code_clean.split("\n") if l.strip()]
            if len(non_empty) < 5:
                continue

            difficulty = assign_difficulty(code_clean)
            scenario_id = f"findings/{protocol_type}/{finding_id}/snippet{block_idx}"
            scenarios.append({
                "id": scenario_id,
                "protocol_type": protocol_type,
                "category_slug": category_slug,
                "category_title": title,
                "canonical_category": canonical,
                "code_clean": code_clean,
                "code_raw": code_raw,
                "hints": [],
                "preconditions": summary,
                "bug_lines": audit_lines,
                "difficulty": difficulty,
                "ground_truth_severity": impact,
                "finding_id": finding_id,
                "protocol_name": protocol_name,
                "firm_name": firm_name,
                "quality_score": quality,
                "source_file": "",
            })

    return scenarios


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_scenarios.py /path/to/protocol-vulnerabilities-index")
        sys.exit(1)

    index_repo = Path(sys.argv[1])
    findings_dir = index_repo / "data" / "findings" / "protocols"
    if not findings_dir.exists():
        print(f"Error: {findings_dir} not found")
        sys.exit(1)

    data_dir = Path(__file__).parent / "data"
    repos_dir = data_dir / "repos"

    if not repos_dir.exists():
        print(f"Warning: {repos_dir} not found. Run fetch_repos.py first.")
        print("Falling back to content-only extraction.\n")

    # Process all findings
    all_scenarios = []
    source_counts = {"repo": 0, "snippet": 0, "skipped": 0}
    total_findings = 0

    for json_file in sorted(findings_dir.glob("*.json")):
        protocol_type = json_file.stem
        data = json.loads(json_file.read_text(encoding="utf-8"))
        total_findings += len(data)

        for finding in data:
            scenarios = process_finding(finding, protocol_type, repos_dir)
            if scenarios:
                for s in scenarios:
                    if s.get("source_file"):
                        source_counts["repo"] += 1
                    else:
                        source_counts["snippet"] += 1
                all_scenarios.extend(scenarios)
            else:
                source_counts["skipped"] += 1

    print(f"Processed {total_findings} findings from {len(list(findings_dir.glob('*.json')))} protocol files")
    print(f"Created {len(all_scenarios)} base scenarios:")
    print(f"  From cloned repos: {source_counts['repo']}")
    print(f"  From content snippets: {source_counts['snippet']}")
    print(f"  Skipped (no code / low quality): {source_counts['skipped']}")

    # Load optional clean / OOD scenarios
    clean_scenarios = _load_optional_scenarios(data_dir / "clean_scenarios.json", "clean")
    ood_scenarios = _load_optional_scenarios(data_dir / "ood_scenarios.json", "ood")

    # Clean scenarios join the main pool (train/eval split).
    # OOD scenarios go into scenarios.json but get their own eval file only.
    all_scenarios.extend(clean_scenarios)
    all_scenarios.extend(ood_scenarios)

    if clean_scenarios:
        print(f"Loaded {len(clean_scenarios)} clean scenarios from data/clean_scenarios.json")
    if ood_scenarios:
        print(f"Loaded {len(ood_scenarios)} OOD scenarios from data/ood_scenarios.json")

    # Stats
    canonical_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    severity_counts = defaultdict(int)
    for s in all_scenarios:
        canonical_counts[s["canonical_category"]] += 1
        difficulty_counts[s["difficulty"]] += 1
        severity_counts[s.get("ground_truth_severity", "UNKNOWN")] += 1

    print(f"\nCanonical categories ({len(canonical_counts)}):")
    for cat, count in sorted(canonical_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulty_counts.items()):
        print(f"  {diff}: {count}")

    print(f"\nSeverity distribution:")
    for sev, count in sorted(severity_counts.items()):
        print(f"  {sev}: {count}")

    # Stratified train/eval split (80/20) — excludes OOD scenarios
    random.seed(42)
    ood_ids = {s["id"] for s in ood_scenarios}
    by_category = defaultdict(list)
    for s in all_scenarios:
        if s["id"] not in ood_ids:
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
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "scenarios.json", "w") as f:
        json.dump(all_scenarios, f, indent=2)
    print(f"Wrote {data_dir / 'scenarios.json'}")

    train_tasks = [build_task_entry(sid) for sid in train_ids]
    with open(data_dir / "tasks_train.json", "w") as f:
        json.dump(train_tasks, f, indent=2)
    print(f"Wrote {data_dir / 'tasks_train.json'} ({len(train_tasks)} tasks)")

    eval_tasks = [build_task_entry(sid) for sid in eval_ids]
    with open(data_dir / "tasks_eval.json", "w") as f:
        json.dump(eval_tasks, f, indent=2)
    print(f"Wrote {data_dir / 'tasks_eval.json'} ({len(eval_tasks)} tasks)")

    if clean_scenarios:
        clean_tasks = [build_task_entry(s["id"]) for s in clean_scenarios]
        with open(data_dir / "tasks_eval_clean.json", "w") as f:
            json.dump(clean_tasks, f, indent=2)
        print(f"Wrote {data_dir / 'tasks_eval_clean.json'} ({len(clean_tasks)} tasks)")

    if ood_scenarios:
        ood_tasks = [build_task_entry(s["id"]) for s in ood_scenarios]
        with open(data_dir / "tasks_eval_ood.json", "w") as f:
            json.dump(ood_tasks, f, indent=2)
        print(f"Wrote {data_dir / 'tasks_eval_ood.json'} ({len(ood_tasks)} tasks)")


if __name__ == "__main__":
    main()
