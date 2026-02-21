"""Solidity Vulnerability Detection — RL Environment.

An agent audits Solidity code snippets and identifies security vulnerabilities.
Built on real DeFi audit data from protocol-vulnerabilities-index (10,600 findings).

Tools:  read_code, get_context, list_hints, submit_finding
Scenario: detect-vuln (setup → agent audits → score)
Scoring: deterministic (no LLM), weighted subscores
"""

import json
import random
import re
from pathlib import Path
from typing import Any

from hud import Environment
from hud.tools.types import EvaluationResult, SubScore

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "data" / "scenarios.json"

_ALL_SCENARIOS: list[dict] = []


def _load_scenarios() -> list[dict]:
    global _ALL_SCENARIOS
    if not _ALL_SCENARIOS:
        with open(DATA_PATH) as f:
            _ALL_SCENARIOS = json.load(f)
    return _ALL_SCENARIOS


# ---------------------------------------------------------------------------
# Canonical category keywords (for fuzzy scoring)
# ---------------------------------------------------------------------------

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
        "unsafe cast", "truncation", "uint128", "int256",
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
# Episode state
# ---------------------------------------------------------------------------

_current: dict = {}
_hints_used: bool = False
_code_read: bool = False

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

env = Environment("solidity-vuln-detect")


@env.tool()
def read_code() -> str:
    """Read the Solidity code snippet under review.

    Returns the smart contract code that may contain a security vulnerability.
    Examine it carefully for common vulnerability patterns like reentrancy,
    access control issues, oracle manipulation, etc."""
    global _code_read
    _code_read = True
    code = _current.get("code_clean", "No code loaded.")
    # Add line numbers for reference
    lines = code.split("\n")
    numbered = "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))
    return numbered


@env.tool()
def get_context() -> str:
    """Get context about the protocol being audited.

    Returns the protocol type and preconditions that describe when this
    vulnerability pattern applies. No reward penalty for using this tool."""
    protocol_type = _current.get("protocol_type", "unknown")
    preconditions = _current.get("preconditions", "No preconditions available.")
    return (
        f"Protocol type: {protocol_type}\n\n"
        f"Preconditions:\n{preconditions}"
    )


@env.tool()
def list_hints() -> str:
    """Request detection heuristics for analyzing this code.

    Returns a list of things to look for when auditing this code.
    WARNING: Using hints reduces your maximum reward from 1.0 to 0.7."""
    global _hints_used
    _hints_used = True
    hints = _current.get("hints", [])
    if not hints:
        return "No hints available for this snippet."
    return "Detection heuristics:\n" + "\n".join(
        f"  {i+1}. {h}" for i, h in enumerate(hints)
    )


@env.tool()
def submit_finding(
    vulnerability_type: str,
    explanation: str,
    severity: str = "HIGH",
    affected_lines: str = "",
    attack_path: str = "",
    prerequisites: str = "",
    impact: str = "",
) -> str:
    """Submit your vulnerability analysis.

    Args:
        vulnerability_type: Category of vulnerability (e.g. 'reentrancy',
            'oracle manipulation', 'access control', 'flash loan',
            'precision loss', 'denial of service', 'frontrunning', etc.)
        explanation: Detailed explanation of WHY the code is vulnerable,
            including what an attacker could exploit and how.
        severity: Impact level — 'HIGH' or 'MEDIUM'.
        affected_lines: Comma-separated line numbers where the root cause is
            (e.g. '5,6,7'). Use the line numbers from read_code().
        attack_path: Step-by-step outline of how the exploit is executed.
        prerequisites: Preconditions required for the exploit to work.
        impact: Expected impact if exploited (e.g. fund loss, stolen assets).
    """
    parsed_lines = []
    if affected_lines.strip():
        for part in affected_lines.split(","):
            part = part.strip()
            if part.isdigit():
                parsed_lines.append(int(part))

    _current["_submission"] = {
        "vulnerability_type": vulnerability_type,
        "explanation": explanation,
        "severity": severity.upper(),
        "affected_lines": parsed_lines,
        "attack_path": attack_path,
        "prerequisites": prerequisites,
        "impact": impact,
    }
    return f"Finding submitted: {vulnerability_type} ({severity}). Awaiting evaluation."


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------

@env.scenario("detect-vuln")
async def detect_vulnerability(scenario_id: str = "") -> Any:
    """Agent must identify the vulnerability in a Solidity code snippet."""
    global _current, _hints_used, _code_read

    scenarios = _load_scenarios()

    if scenario_id:
        matches = [s for s in scenarios if s["id"] == scenario_id]
        if not matches:
            raise ValueError(f"Unknown scenario_id: {scenario_id}")
        _current = matches[0].copy()
    else:
        _current = random.choice(scenarios).copy()

    _hints_used = False
    _code_read = False
    _current["_submission"] = None

    protocol_type = _current["protocol_type"]

    # First yield: prompt
    answer = yield (
        f"You are a smart contract security auditor reviewing a {protocol_type} protocol.\n\n"
        f"Instructions:\n"
        f"1. Use read_code() to examine the Solidity code snippet\n"
        f"2. Optionally use get_context() for protocol details\n"
        f"3. Optionally use list_hints() for detection guidance (reduces max reward)\n"
        f"4. Use submit_finding() with your analysis when ready\n\n"
        f"Your goal: identify the specific vulnerability type, explain the attack vector, "
        f"and pinpoint the affected lines of code."
    )

    # Second yield: evaluation
    submission = _current.get("_submission")
    if not submission:
        yield EvaluationResult(
            reward=0.0,
            done=True,
            content="No finding was submitted.",
            info={"scenario_id": _current["id"]},
        )
        return

    # Score components
    cat_score = _score_category(
        submission["vulnerability_type"],
        _current["category_slug"],
        _current["canonical_category"],
    )
    expl_score = _score_explanation(submission["explanation"], _current)
    sev_score = _score_severity(submission["severity"], _current)
    line_score = _score_lines(submission["affected_lines"], _current.get("bug_lines", []))
    exploit_score = _score_exploitability(
        submission.get("attack_path", ""),
        submission.get("prerequisites", ""),
        submission.get("impact", ""),
    )

    # Weighted combination
    raw = (
        0.45 * cat_score
        + 0.25 * expl_score
        + 0.10 * sev_score
        + 0.15 * line_score
        + 0.05 * exploit_score
    )
    hint_penalty = 0.7 if _hints_used else 1.0
    final = round(raw * hint_penalty, 4)

    yield EvaluationResult(
        reward=final,
        done=True,
        content=(
            f"Submitted: {submission['vulnerability_type']} "
            f"(expected: {_current['canonical_category']})\n"
            f"Scores — category: {cat_score:.2f}, explanation: {expl_score:.2f}, "
            f"severity: {sev_score:.2f}, lines: {line_score:.2f}, "
            f"exploitability: {exploit_score:.2f}\n"
            f"Hint penalty: {hint_penalty}, Final reward: {final}"
        ),
        subscores=[
            SubScore(name="category_match", weight=0.45, value=cat_score),
            SubScore(name="explanation_quality", weight=0.25, value=expl_score),
            SubScore(name="severity_match", weight=0.10, value=sev_score),
            SubScore(name="line_accuracy", weight=0.15, value=line_score),
            SubScore(name="exploitability", weight=0.05, value=exploit_score),
        ],
        info={
            "scenario_id": _current["id"],
            "canonical_category": _current["canonical_category"],
            "difficulty": _current["difficulty"],
            "hints_used": _hints_used,
            "code_read": _code_read,
            "submitted_type": submission["vulnerability_type"],
            "submitted_severity": submission["severity"],
            "submitted_lines": submission["affected_lines"],
            "submitted_attack_path": submission.get("attack_path", ""),
            "submitted_prerequisites": submission.get("prerequisites", ""),
            "submitted_impact": submission.get("impact", ""),
            "ground_truth_lines": _current.get("bug_lines", []),
        },
    )


# ---------------------------------------------------------------------------
# Scoring functions (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _to_words(text: str) -> set[str]:
    """Normalize text to a set of lowercase words."""
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _score_category(submitted: str, ground_slug: str, ground_canonical: str) -> float:
    """Score vulnerability type identification."""
    sub = submitted.lower().strip()
    sub_slug = re.sub(r"[^a-z0-9]+", "-", sub).strip("-")
    sub_words = _to_words(submitted)

    # --- Exact match on canonical or raw slug ---
    if sub_slug == ground_canonical or sub_slug == ground_slug:
        return 1.0

    # --- Word containment: all canonical words appear in submission ---
    canonical_words = _to_words(ground_canonical)
    if canonical_words and canonical_words.issubset(sub_words):
        return 1.0

    # Raw slug words appear in submission
    slug_words = _to_words(ground_slug)
    if slug_words and slug_words.issubset(sub_words):
        return 1.0

    # --- Check canonical group membership ---
    from build_scenarios import CANONICAL_CATEGORIES
    if ground_canonical in CANONICAL_CATEGORIES:
        for slug in CANONICAL_CATEGORIES[ground_canonical]:
            if sub_slug == slug:
                return 1.0
            # Word overlap: if slug words are a subset of submission
            if _to_words(slug).issubset(sub_words):
                return 1.0

    # --- Keyword match (lowered thresholds) ---
    if ground_canonical in CATEGORY_KEYWORDS:
        keywords = CATEGORY_KEYWORDS[ground_canonical]
        hits = sum(1 for kw in keywords if kw.lower() in sub)
        if hits >= 2:
            return 0.9
        if hits >= 1:
            return 0.6

    # --- Jaccard similarity on words ---
    if canonical_words:
        jaccard = len(sub_words & canonical_words) / len(sub_words | canonical_words) if sub_words else 0
        if jaccard >= 0.5:
            return 0.8
        if jaccard > 0:
            return 0.4

    # --- Fallback: any significant canonical word in submission ---
    for w in canonical_words:
        if len(w) > 4 and w in sub:
            return 0.4

    # --- Fallback: any significant slug word in submission ---
    for w in slug_words:
        if len(w) > 4 and w in sub:
            return 0.3

    return 0.0


def _score_explanation(explanation: str, scenario: dict) -> float:
    """Score explanation quality via keyword heuristics."""
    expl = explanation.lower()
    score = 0.0

    # 1. References code identifiers (0.0 - 0.3)
    code = scenario.get("code_raw", "")
    identifiers = set(re.findall(r"\b([a-zA-Z_]\w{2,})\b", code))
    solidity_kw = {
        "function", "external", "internal", "public", "private", "view",
        "returns", "require", "uint256", "address", "bool", "mapping",
        "memory", "storage", "calldata", "return", "emit", "event",
        "modifier", "contract", "import", "pragma", "solidity", "msg",
        "sender", "this", "true", "false", "string", "bytes", "uint",
        "int256", "bytes32", "the", "and", "for", "not", "new",
    }
    meaningful = identifiers - solidity_kw
    if meaningful:
        mentioned = sum(1 for ident in meaningful if ident.lower() in expl)
        score += min(0.3, 0.1 * mentioned)

    # 2. Attack vector language (0.0 - 0.4)
    attack_terms = [
        "attacker", "exploit", "manipulate", "steal", "drain", "bypass",
        "frontrun", "front-run", "sandwich", "flash loan", "reenter",
        "reentrancy", "inflate", "vulnerable", "vulnerability",
        "because", "since", "allows", "enables", "leads to", "resulting",
        "before", "after", "during", "sequence", "ordering",
        "profit", "loss", "funds", "tokens", "ETH", "shares",
        "malicious", "unauthorized", "incorrect", "missing",
    ]
    hits = sum(1 for t in attack_terms if t.lower() in expl)
    score += min(0.4, 0.10 * hits)

    # 3. Sufficient length (0.0 - 0.15)
    words = len(explanation.split())
    if words >= 50:
        score += 0.15
    elif words >= 25:
        score += 0.15
    elif words >= 10:
        score += 0.10
    elif words >= 5:
        score += 0.05

    # 4. References preconditions (0.0 - 0.15)
    precond = scenario.get("preconditions", "").lower()
    if precond:
        precond_words = set(re.findall(r"\b\w{4,}\b", precond))
        overlap = sum(1 for w in precond_words if w in expl)
        score += min(0.15, 0.03 * overlap)

    return min(1.0, score)


def _score_severity(submitted: str, scenario: dict) -> float:
    """Score severity classification. Participation-based (no reliable ground truth)."""
    sub = submitted.upper().strip()
    if sub in ("HIGH", "MEDIUM", "LOW", "CRITICAL"):
        return 0.8  # Credit for providing a valid severity
    return 0.3  # Submitted something but not a standard severity


def _score_lines(submitted: list[int], ground_truth: list[int]) -> float:
    """Score affected line identification via overlap."""
    if not ground_truth:
        # No ground truth — give flat 0.5 (no noise from unannotated scenarios)
        return 0.5

    if not submitted:
        return 0.0

    gt_set = set(ground_truth)
    sub_set = set(submitted)

    # Allow ±1 line tolerance
    expanded_gt = set()
    for line in gt_set:
        expanded_gt.update([line - 1, line, line + 1])

    hits = len(sub_set & expanded_gt)
    precision = hits / len(sub_set) if sub_set else 0
    recall = hits / len(gt_set) if gt_set else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def _score_exploitability(attack_path: str, prerequisites: str, impact: str) -> float:
    """Score exploitability fields based on completeness."""
    def field_score(text: str) -> float:
        words = len(text.split())
        if words >= 8:
            return 1.0
        if words >= 4:
            return 0.7
        if words >= 1:
            return 0.4
        return 0.0

    attack_score = field_score(attack_path)
    prereq_score = field_score(prerequisites)
    impact_score = field_score(impact)
    score = 0.4 * attack_score + 0.3 * prereq_score + 0.3 * impact_score
    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Run as MCP server or test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env.run(transport="stdio")
