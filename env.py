"""Solidity Vulnerability Detection — RL Environment.

An agent audits Solidity code snippets and identifies security vulnerabilities.
Built on real DeFi audit data from protocol-vulnerabilities-index (10,600 findings).

Tools:  read_code, get_context, list_hints, submit_finding
Scenario: detect-vuln (setup → agent audits → score)
Scoring: deterministic (no LLM), weighted subscores
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

_HUD_AVAILABLE = False
try:
    from hud import Environment
    from hud.tools.types import EvaluationResult, SubScore
    _HUD_AVAILABLE = True
except ImportError:
    pass

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


def _parse_affected_lines(value: str | list[int] | list[str] | None) -> list[int]:
    """Parse line specs like ``10,12,14`` and ``10-14`` into sorted ints."""
    if value is None:
        return []

    if isinstance(value, list):
        parsed: list[int] = []
        for item in value:
            parsed.extend(_parse_affected_lines(item))
        return sorted(set(parsed))

    text = str(value).strip()
    if not text:
        return []

    parsed = set()
    for token in re.findall(r"\d+\s*-\s*\d+|\d+", text):
        if "-" not in token:
            parsed.add(int(token))
            continue

        start_text, end_text = re.split(r"\s*-\s*", token, maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            start, end = end, start
        if end - start > 200:
            end = start + 200
        parsed.update(range(start, end + 1))

    return sorted(parsed)


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
    "no-vulnerability": [
        "no vulnerability", "no-vulnerability", "safe", "clean",
        "no issue", "secure", "not vulnerable", "no bug", "no finding",
    ],
}

# ---------------------------------------------------------------------------
# Expected severity per canonical category
# ---------------------------------------------------------------------------

CATEGORY_SEVERITY: dict[str, str] = {
    "reentrancy": "HIGH",
    "oracle-manipulation": "HIGH",
    "access-control": "HIGH",
    "flash-loan": "HIGH",
    "first-depositor-inflation": "MEDIUM",
    "precision-rounding": "MEDIUM",
    "slippage-protection": "MEDIUM",
    "fee-on-transfer": "MEDIUM",
    "integer-overflow": "HIGH",
    "denial-of-service": "MEDIUM",
    "frontrunning-mev": "MEDIUM",
    "governance": "MEDIUM",
    "liquidation": "HIGH",
    "input-validation": "LOW",
    "reward-accounting": "MEDIUM",
    "unchecked-returns": "MEDIUM",
    "initialization": "HIGH",
    "erc4626-vault": "MEDIUM",
    "locked-funds": "MEDIUM",
    "stale-state": "MEDIUM",
    "signature-replay": "HIGH",
    "incorrect-math": "MEDIUM",
    "no-vulnerability": "NONE",
}

_SEVERITY_RANK: dict[str, int] = {
    "NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4,
}

# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

_current: dict = {}
_hints_used: bool = False
_code_read: bool = False

# ---------------------------------------------------------------------------
# HUD Environment (optional — only registered when hud SDK is installed)
# ---------------------------------------------------------------------------

if _HUD_AVAILABLE:
    env = Environment("solidity-vuln-detect")

    @env.tool()
    def read_code() -> str:
        """Read the Solidity code snippet under review."""
        global _code_read
        _code_read = True
        code = _current.get("code_clean", "No code loaded.")
        lines = code.split("\n")
        return "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))

    @env.tool()
    def get_context() -> str:
        """Get context about the protocol being audited."""
        protocol_type = _current.get("protocol_type", "unknown")
        protocol_name = _current.get("protocol_name", "")
        preconditions = _current.get("preconditions", "No preconditions available.")
        header = f"Protocol type: {protocol_type}"
        if protocol_name:
            header += f"\nProtocol: {protocol_name}"
        return f"{header}\n\nPreconditions:\n{preconditions}"

    @env.tool()
    def list_hints() -> str:
        """Request detection heuristics. WARNING: reduces max reward to 0.7."""
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
        """Submit your vulnerability analysis."""
        _current["_submission"] = {
            "vulnerability_type": vulnerability_type,
            "explanation": explanation,
            "severity": severity.upper(),
            "affected_lines": _parse_affected_lines(affected_lines),
            "attack_path": attack_path,
            "prerequisites": prerequisites,
            "impact": impact,
        }
        return f"Finding submitted: {vulnerability_type} ({severity}). Awaiting evaluation."

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
        submission = _current.get("_submission")
        if not submission:
            yield EvaluationResult(
                reward=0.0, done=True,
                content="No finding was submitted.",
                info={"scenario_id": _current["id"]},
            )
            return
        scored = evaluate_submission(
            _current,
            submission,
            hints_used=_hints_used,
            code_read=_code_read,
        )
        yield EvaluationResult(
            reward=scored["reward"], done=True,
            content=scored["content"],
            subscores=[
                SubScore(name="category_match", weight=0.40, value=scored["subscores"]["category_match"]),
                SubScore(name="explanation_quality", weight=0.25, value=scored["subscores"]["explanation_quality"]),
                SubScore(name="severity_match", weight=0.10, value=scored["subscores"]["severity_match"]),
                SubScore(name="line_accuracy", weight=0.15, value=scored["subscores"].get("line_accuracy") or 0.0),
                SubScore(name="exploitability", weight=0.10, value=scored["subscores"]["exploitability"]),
            ],
            info=scored["info"],
        )


# ---------------------------------------------------------------------------
# Scoring functions (deterministic, no LLM — always available)
# ---------------------------------------------------------------------------

def _to_words(text: str) -> set[str]:
    """Normalize text to a set of lowercase words."""
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _score_category(submitted: str, ground_slug: str, ground_canonical: str) -> float:
    """Score vulnerability type identification."""
    sub = submitted.lower().strip()
    sub_slug = re.sub(r"[^a-z0-9]+", "-", sub).strip("-")
    sub_words = _to_words(submitted)

    # --- Hard rules for "no-vulnerability" class ---
    _NO_VULN_SLUGS = {
        "no-vulnerability", "no-vulnerabilities", "no-vuln",
        "safe", "clean", "secure", "no-issue", "no-issues",
        "no-bug", "no-bugs", "no-finding", "no-findings",
        "not-vulnerable",
    }
    sub_is_clean = sub_slug in _NO_VULN_SLUGS or "no vulnerability" in sub or "no-vulnerability" in sub
    ground_is_clean = ground_canonical == "no-vulnerability"

    if ground_is_clean and sub_is_clean:
        return 1.0
    if ground_is_clean and not sub_is_clean:
        return 0.0  # False positive: reported vuln on safe code
    if not ground_is_clean and sub_is_clean:
        return 0.0  # False negative: said "safe" on vulnerable code

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
            return 0.8

    # --- Jaccard similarity on words ---
    if canonical_words:
        jaccard = len(sub_words & canonical_words) / len(sub_words | canonical_words) if sub_words else 0
        if jaccard >= 0.5:
            return 0.8
        if jaccard > 0:
            return 0.5

    # --- Fallback: any significant canonical word in submission ---
    for w in canonical_words:
        if len(w) > 4 and w in sub:
            return 0.4

    # --- Fallback: any significant slug word in submission ---
    for w in slug_words:
        if len(w) > 4 and w in sub:
            return 0.3

    # --- Cross-category: submission matches a real category (wrong one) ---
    for other_cat, other_kws in CATEGORY_KEYWORDS.items():
        if other_cat == ground_canonical or other_cat == "no-vulnerability":
            continue
        if sum(1 for kw in other_kws if kw.lower() in sub) >= 2:
            return 0.3  # Found a real vuln category, just wrong one

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
        score += 0.12
    elif words >= 10:
        score += 0.08
    elif words >= 5:
        score += 0.04

    # 4. References preconditions (0.0 - 0.15)
    precond = scenario.get("preconditions", "").lower()
    if precond:
        precond_words = set(re.findall(r"\b\w{4,}\b", precond))
        overlap = sum(1 for w in precond_words if w in expl)
        score += min(0.15, 0.03 * overlap)

    return min(1.0, score)


def _score_severity(submitted: str, scenario: dict) -> float:
    """Score severity classification against expected severity for the category."""
    sub = submitted.upper().strip()
    canonical = scenario.get("canonical_category", "")
    expected = scenario.get("ground_truth_severity") or CATEGORY_SEVERITY.get(canonical)

    if expected is None:
        # Unknown category — fall back to participation scoring
        return 0.8 if sub in _SEVERITY_RANK else 0.1

    if sub == expected:
        return 1.0

    sub_rank = _SEVERITY_RANK.get(sub)
    if sub_rank is None:
        return 0.1

    distance = abs(sub_rank - _SEVERITY_RANK[expected])
    if distance == 1:
        return 0.6
    if distance == 2:
        return 0.3
    return 0.1


def _score_lines(submitted: list[int], ground_truth: list[int]) -> float | None:
    """Score affected line identification via overlap."""
    if not ground_truth:
        return None

    if not submitted:
        return 0.0

    gt_set = set(ground_truth)
    sub_set = set(submitted)

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


def evaluate_submission(
    scenario: dict,
    submission: dict | None,
    *,
    hints_used: bool = False,
    code_read: bool = False,
) -> dict:
    """Score a submission against a scenario using the shared reward logic."""
    if not submission:
        return {
            "reward": 0.0,
            "subscores": {},
            "info": {"scenario_id": scenario["id"], "error": "no_submission"},
            "content": "No finding was submitted.",
        }

    cat_score = _score_category(
        submission["vulnerability_type"],
        scenario["category_slug"],
        scenario["canonical_category"],
    )
    expl_score = _score_explanation(submission["explanation"], scenario)
    sev_score = _score_severity(submission["severity"], scenario)
    line_score = _score_lines(
        submission["affected_lines"],
        scenario.get("bug_lines", []),
    )
    exploit_score = _score_exploitability(
        submission.get("attack_path", ""),
        submission.get("prerequisites", ""),
        submission.get("impact", ""),
        ground_canonical=scenario["canonical_category"],
    )

    schema_penalty = 1.0
    if scenario["canonical_category"] != "no-vulnerability":
        if all(not submission.get(f, "").strip() for f in ("attack_path", "prerequisites", "impact")):
            schema_penalty *= 0.9
    if "," in submission["vulnerability_type"]:
        schema_penalty *= 0.85

    weighted_scores = {
        "category_match": (0.40, cat_score),
        "explanation_quality": (0.25, expl_score),
        "severity_match": (0.10, sev_score),
        "line_accuracy": (0.15, line_score),
        "exploitability": (0.10, exploit_score),
    }
    numerator = 0.0
    denominator = 0.0
    for weight, score in weighted_scores.values():
        if score is None:
            continue
        numerator += weight * score
        denominator += weight
    raw = numerator / denominator if denominator else 0.0

    hint_penalty = 0.7 if hints_used else 1.0
    final = round(raw * schema_penalty * hint_penalty, 4)
    line_display = "n/a" if line_score is None else f"{line_score:.2f}"

    return {
        "reward": final,
        "subscores": {
            "category_match": round(cat_score, 4),
            "explanation_quality": round(expl_score, 4),
            "severity_match": round(sev_score, 4),
            "line_accuracy": None if line_score is None else round(line_score, 4),
            "exploitability": round(exploit_score, 4),
        },
        "info": {
            "scenario_id": scenario["id"],
            "canonical_category": scenario["canonical_category"],
            "difficulty": scenario.get("difficulty", "unknown"),
            "hints_used": hints_used,
            "code_read": code_read,
            "schema_penalty": schema_penalty,
            "submitted_type": submission["vulnerability_type"],
            "submitted_severity": submission["severity"],
            "submitted_lines": submission["affected_lines"],
            "ground_truth_lines": scenario.get("bug_lines", []),
            "line_accuracy_scored": line_score is not None,
        },
        "content": (
            f"Submitted: {submission['vulnerability_type']} "
            f"(expected: {scenario['canonical_category']})\n"
            f"Scores — cat: {cat_score:.2f}, expl: {expl_score:.2f}, "
            f"sev: {sev_score:.2f}, lines: {line_display}, "
            f"exploit: {exploit_score:.2f}\n"
            f"Final reward: {final}"
        ),
    }


def _score_exploitability(
    attack_path: str,
    prerequisites: str,
    impact: str,
    ground_canonical: str = "",
) -> float:
    """Score exploitability fields based on completeness and quality."""
    # --- No-vulnerability special case ---
    if ground_canonical == "no-vulnerability":
        all_empty = all(
            t.strip().lower() in ("", "n/a", "none", "not applicable")
            for t in (attack_path, prerequisites, impact)
        )
        if all_empty:
            return 1.0
        total_words = sum(len(t.split()) for t in (attack_path, prerequisites, impact))
        if total_words > 10:
            return 0.2  # Heavy penalty for detailed false positive exploit
        return 0.5

    # --- Quality keyword sets ---
    _ACTION_VERBS = {
        "call", "calls", "transfer", "transfers", "deploy", "deploys",
        "invoke", "invokes", "manipulate", "manipulates", "drain", "drains",
        "execute", "executes", "send", "sends", "approve", "approves",
        "borrow", "borrows", "swap", "swaps", "withdraw", "withdraws",
        "deposit", "deposits", "mint", "mints", "burn", "burns",
        "exploit", "exploits", "trigger", "triggers",
    }
    _SEQUENCING = {
        "step", "then", "first", "next", "finally", "subsequently",
        "after", "before", "1.", "2.", "3.", "4.", "5.",
    }
    _IMPACT_TERMS = {
        "funds", "tokens", "eth", "ether", "drain", "steal", "lock",
        "dos", "loss", "profit", "damage", "collateral", "liquidat",
        "insolvency", "bad debt", "freeze", "stuck", "permanently",
    }
    _PREREQ_TERMS = {
        "requires", "must", "needs", "assumes", "when", "if",
        "given", "condition", "prerequisite", "necessary",
    }

    def _field_score(text: str, quality_terms: set[str]) -> float:
        text_lower = text.lower()
        words = len(text.split())

        if words >= 8:
            baseline = 1.0
        elif words >= 4:
            baseline = 0.7
        elif words >= 1:
            baseline = 0.4
        else:
            return 0.0

        hits = sum(1 for term in quality_terms if term in text_lower)
        if hits >= 3:
            quality = 1.0
        elif hits >= 2:
            quality = 0.9
        elif hits >= 1:
            quality = 0.75
        else:
            quality = 0.5

        return baseline * quality

    attack_score = _field_score(attack_path, _ACTION_VERBS | _SEQUENCING)
    prereq_score = _field_score(prerequisites, _PREREQ_TERMS)
    impact_score = _field_score(impact, _IMPACT_TERMS)
    score = 0.4 * attack_score + 0.3 * prereq_score + 0.3 * impact_score
    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# Run as MCP server or test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not _HUD_AVAILABLE:
        raise SystemExit(
            "hud-python is not installed; env.py only exposes scoring helpers in standalone mode."
        )
    env.run(transport="stdio")
