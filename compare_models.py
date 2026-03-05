"""Deep-dive comparison of model evaluation results.

Loads result files for multiple models and produces a comprehensive
side-by-side analysis: overall stats, subscores, per-category,
per-difficulty, head-to-head, error analysis, and tool usage.

Usage:
    python compare_models.py
    python compare_models.py --brief
    python compare_models.py --results results/model_a.json results/model_b.json
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(vals):
    return statistics.mean(vals) if vals else 0.0

def _median(vals):
    return statistics.median(vals) if vals else 0.0

def _stdev(vals):
    return statistics.stdev(vals) if len(vals) > 1 else 0.0

def _pct(n, total):
    return f"{n/total*100:.1f}%" if total else "N/A"

def _bar(val, width=20):
    filled = round(val * width)
    return "█" * filled + "░" * (width - filled)

def _short_model(name):
    name = name.replace("claude-", "").replace("-20251001", "")
    return name.capitalize() if len(name) < 12 else name

def _correlation(xs, ys):
    """Pearson correlation coefficient."""
    if len(xs) < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    sx, sy = _stdev(xs), _stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    n = len(xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((n - 1) * sx * sy)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(paths):
    """Load result files, return list of (model_name, summary_dict, traces_list)."""
    models = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        model = data["model"]
        traces = data.get("traces", [])
        models.append((model, data, traces))

    # Deduplicate: if the same model name appears more than once, append date
    name_counts = defaultdict(int)
    for m, _, _ in models:
        name_counts[m] += 1
    if any(c > 1 for c in name_counts.values()):
        deduped = []
        for name, data, traces in models:
            if name_counts[name] > 1:
                ts = data.get("timestamp", "")[:10]
                name = f"{name} ({ts})"
            deduped.append((name, data, traces))
        models = deduped

    return models


def build_scenario_index(all_models):
    """Build {scenario_id: {model: trace}} for head-to-head comparison."""
    index = defaultdict(dict)
    for model, _, traces in all_models:
        for t in traces:
            sid = t.get("scenario_id") or t.get("info", {}).get("scenario_id", "")
            if sid and t.get("error") is None:
                index[sid][model] = t
    return index


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_overall(all_models):
    print("=" * 90)
    print("  1. OVERALL SUMMARY")
    print("=" * 90)
    print()

    # Header
    labels = [_short_model(m) for m, _, _ in all_models]
    header = f"{'Metric':<22}" + "".join(f"{l:>20}" for l in labels)
    print(header)
    print("-" * len(header))

    # Rows
    rows = [
        ("Completed", lambda d, t: f"{d['completed']}/{d['total_tasks']}"),
        ("Errors", lambda d, t: str(d.get("errors", 0))),
        ("Mean Reward", lambda d, t: f"{d['mean_reward']:.4f}"),
        ("Median Reward", lambda d, t: f"{d['median_reward']:.4f}"),
        ("Stdev", lambda d, t: f"{d['stdev_reward']:.4f}"),
        ("Min Reward", lambda d, t: f"{min(x['reward'] for x in t if x.get('error') is None):.4f}"),
        ("Max Reward", lambda d, t: f"{max(x['reward'] for x in t if x.get('error') is None):.4f}"),
    ]
    for label, fn in rows:
        vals = [fn(d, t) for _, d, t in all_models]
        print(f"{label:<22}" + "".join(f"{v:>20}" for v in vals))

    print()


def section_subscores(all_models):
    print("=" * 90)
    print("  2. SUBSCORE DEEP-DIVE")
    print("=" * 90)
    print()

    subscore_names = ["category_match", "explanation_quality", "severity_match",
                      "line_accuracy", "exploitability"]
    weights = {"category_match": 0.40, "explanation_quality": 0.25,
               "severity_match": 0.10, "line_accuracy": 0.15, "exploitability": 0.10}

    # Compute means
    model_subscores = {}
    for model, _, traces in all_models:
        model_subscores[model] = {}
        for s in subscore_names:
            vals = [t["subscores"].get(s, 0.0) for t in traces if t.get("error") is None and t.get("subscores")]
            model_subscores[model][s] = _mean(vals)

    labels = [_short_model(m) for m, _, _ in all_models]
    header = f"{'Subscore':<25}{'Weight':>7}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    for s in subscore_names:
        w = weights[s]
        vals = [f"{model_subscores[m][s]:.4f}" for m, _, _ in all_models]
        print(f"{s:<25}{w:>6.0%}" + "".join(f"{v:>15}" for v in vals))

    print()

    # Correlation: which subscore correlates most with final reward?
    print("Subscore-Reward Correlation (how much each subscore drives final reward):")
    print()
    for model, _, traces in all_models:
        good_traces = [t for t in traces if t.get("error") is None and t.get("subscores")]
        rewards = [t["reward"] for t in good_traces]
        print(f"  {_short_model(model)}:")
        corrs = []
        for s in subscore_names:
            vals = [t["subscores"].get(s, 0.0) for t in good_traces]
            r = _correlation(vals, rewards)
            corrs.append((s, r))
        corrs.sort(key=lambda x: -abs(x[1]))
        for s, r in corrs:
            print(f"    {s:<25} r = {r:+.3f}  {_bar(abs(r), 15)}")
        print()


def section_difficulty(all_models):
    print("=" * 90)
    print("  3. BY DIFFICULTY")
    print("=" * 90)
    print()

    difficulties = ["easy", "medium", "hard"]
    subscore_names = ["category_match", "explanation_quality", "severity_match",
                      "line_accuracy", "exploitability"]

    # Aggregate
    model_diff = {}
    for model, _, traces in all_models:
        model_diff[model] = {}
        for diff in difficulties:
            dt = [t for t in traces if t.get("error") is None
                  and t.get("info", {}).get("difficulty") == diff]
            model_diff[model][diff] = {
                "count": len(dt),
                "mean": _mean([t["reward"] for t in dt]),
                "subscores": {s: _mean([t["subscores"].get(s, 0) for t in dt if t.get("subscores")])
                              for s in subscore_names},
            }

    # Main table
    labels = [_short_model(m) for m, _, _ in all_models]
    header = f"{'Difficulty':<12}{'Count':>7}" + "".join(f"{l:>15}" for l in labels) + f"{'Spread':>10}"
    print(header)
    print("-" * len(header))

    for diff in difficulties:
        count = model_diff[all_models[0][0]][diff]["count"]
        means = [model_diff[m][diff]["mean"] for m, _, _ in all_models]
        spread = max(means) - min(means)
        vals = [f"{v:.4f}" for v in means]
        print(f"{diff:<12}{count:>7}" + "".join(f"{v:>15}" for v in vals) + f"{spread:>10.4f}")

    # Easy-Hard gap
    print()
    print("Easy-Hard Gap (how much performance drops on harder tasks):")
    for model, _, _ in all_models:
        easy = model_diff[model]["easy"]["mean"]
        hard = model_diff[model]["hard"]["mean"]
        gap = easy - hard
        print(f"  {_short_model(model):<20} {gap:+.4f}  (easy {easy:.3f} → hard {hard:.3f})")

    # Subscore breakdown by difficulty
    print()
    print("Subscores by Difficulty:")
    for diff in difficulties:
        print(f"\n  [{diff.upper()}]")
        header = f"  {'Subscore':<25}" + "".join(f"{_short_model(m):>15}" for m, _, _ in all_models)
        print(header)
        for s in subscore_names:
            vals = [f"{model_diff[m][diff]['subscores'][s]:.3f}" for m, _, _ in all_models]
            print(f"  {s:<25}" + "".join(f"{v:>15}" for v in vals))

    print()


def section_categories(all_models):
    print("=" * 90)
    print("  4. BY CATEGORY")
    print("=" * 90)
    print()

    # Aggregate by category
    model_cat = {}
    all_cats = set()
    for model, _, traces in all_models:
        model_cat[model] = defaultdict(list)
        for t in traces:
            if t.get("error") is not None:
                continue
            cat = t.get("info", {}).get("canonical_category", "unknown")
            model_cat[model][cat].append(t["reward"])
            all_cats.add(cat)

    cats = sorted(all_cats)
    labels = [_short_model(m) for m, _, _ in all_models]

    # Full table sorted by best model performance
    def best_mean(cat):
        return max(_mean(model_cat[m][cat]) for m, _, _ in all_models)

    cats_sorted = sorted(cats, key=best_mean, reverse=True)

    header = f"{'Category':<32}{'N':>5}" + "".join(f"{l:>12}" for l in labels) + f"{'Spread':>9}"
    print(header)
    print("-" * len(header))

    cat_spreads = {}
    for cat in cats_sorted:
        n = max(len(model_cat[m][cat]) for m, _, _ in all_models)
        means = [_mean(model_cat[m][cat]) for m, _, _ in all_models]
        spread = max(means) - min(means) if means else 0
        cat_spreads[cat] = (spread, means)
        vals = [f"{v:.4f}" if model_cat[m][cat] else "   —" for m, v in zip([m for m, _, _ in all_models], means)]
        print(f"{cat:<32}{n:>5}" + "".join(f"{v:>12}" for v in vals) + f"{spread:>9.3f}")

    # Biggest divergences
    print()
    print("Biggest Model Divergences (categories where models disagree most):")
    divergent = sorted(cat_spreads.items(), key=lambda x: -x[1][0])[:10]
    for cat, (spread, means) in divergent:
        best_idx = means.index(max(means))
        worst_idx = means.index(min(means))
        print(f"  {cat:<30} spread={spread:.3f}  "
              f"best={labels[best_idx]} ({means[best_idx]:.3f})  "
              f"worst={labels[worst_idx]} ({means[worst_idx]:.3f})")

    # Universal strengths and weaknesses
    print()
    print("Universal Strengths (all models > 0.6):")
    for cat in cats_sorted:
        means = [_mean(model_cat[m][cat]) for m, _, _ in all_models]
        if all(v > 0.6 for v in means):
            avg = _mean(means)
            print(f"  {cat:<30} avg={avg:.3f}  ({', '.join(f'{v:.3f}' for v in means)})")

    print()
    print("Universal Weaknesses (all models < 0.5):")
    for cat in cats_sorted:
        means = [_mean(model_cat[m][cat]) for m, _, _ in all_models]
        if all(v < 0.5 for v in means) and any(model_cat[m][cat] for m, _, _ in all_models):
            avg = _mean(means)
            print(f"  {cat:<30} avg={avg:.3f}  ({', '.join(f'{v:.3f}' for v in means)})")

    # Per-model wins
    print()
    for idx, (model, _, _) in enumerate(all_models):
        others_idx = [i for i in range(len(all_models)) if i != idx]
        wins = []
        for cat in cats:
            my_mean = _mean(model_cat[model][cat])
            if not model_cat[model][cat]:
                continue
            other_means = [_mean(model_cat[all_models[i][0]][cat]) for i in others_idx]
            if all(my_mean > om + 0.05 for om in other_means):
                margin = my_mean - max(other_means)
                wins.append((cat, my_mean, margin))
        wins.sort(key=lambda x: -x[2])
        print(f"{labels[idx]}'s Exclusive Wins (> 0.05 ahead of all others):")
        if wins:
            for cat, val, margin in wins[:8]:
                print(f"  {cat:<30} {val:.3f}  (+{margin:.3f})")
        else:
            print("  (none)")
        print()


def section_head_to_head(all_models, scenario_index):
    print("=" * 90)
    print("  5. HEAD-TO-HEAD PER-SCENARIO")
    print("=" * 90)
    print()

    model_names = [m for m, _, _ in all_models]
    labels = [_short_model(m) for m in model_names]

    # Only scenarios where all models succeeded
    common = {sid: d for sid, d in scenario_index.items() if all(m in d for m in model_names)}
    print(f"Common scenarios (all models completed): {len(common)}")
    print()

    # Win counts
    wins = defaultdict(int)
    ties = 0
    margins = defaultdict(list)
    for sid, traces in common.items():
        rewards = {m: traces[m]["reward"] for m in model_names}
        best_reward = max(rewards.values())
        winners = [m for m, r in rewards.items() if r == best_reward]
        if len(winners) > 1:
            ties += 1
        else:
            wins[winners[0]] += 1
            margin = best_reward - sorted(rewards.values())[-2]
            margins[winners[0]].append(margin)

    header = f"{'Model':<20}{'Wins':>8}{'Win %':>10}{'Avg Margin':>14}{'Ties':>8}"
    print(header)
    print("-" * len(header))
    for model in model_names:
        w = wins[model]
        pct = w / len(common) * 100 if common else 0
        avg_m = _mean(margins[model]) if margins[model] else 0
        print(f"{_short_model(model):<20}{w:>8}{pct:>9.1f}%{avg_m:>14.4f}{'':>8}")
    print(f"{'Ties':<20}{ties:>8}{ties/len(common)*100:>9.1f}%")

    # Score thresholds
    print()
    print("Score Distribution:")
    thresholds = [(0.9, "Excellent (≥0.9)"), (0.7, "Good (≥0.7)"), (0.5, "Passing (≥0.5)"), (0.3, "Poor (≥0.3)"), (0.0, "Failed (<0.3)")]
    header = f"{'Tier':<22}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    for i, (thresh, tier_name) in enumerate(thresholds):
        counts = []
        for model in model_names:
            upper = thresholds[i - 1][0] if i > 0 else float('inf')
            if i < len(thresholds) - 1:
                c = sum(1 for sid, d in common.items()
                        if thresh <= d[model]["reward"] < upper)
            else:
                c = sum(1 for sid, d in common.items() if d[model]["reward"] < 0.3)
            counts.append(c)
        vals = [f"{c} ({c/len(common)*100:.1f}%)" for c in counts]
        print(f"{tier_name:<22}" + "".join(f"{v:>15}" for v in vals))

    # Most contested scenarios
    print()
    print("Most Contested Scenarios (highest reward variance across models):")
    contested = []
    for sid, traces in common.items():
        rewards = [traces[m]["reward"] for m in model_names]
        var = statistics.variance(rewards) if len(rewards) > 1 else 0
        contested.append((sid, rewards, var))
    contested.sort(key=lambda x: -x[2])

    header = f"{'Scenario':<45}" + "".join(f"{l:>12}" for l in labels)
    print(header)
    print("-" * len(header))
    for sid, rewards, var in contested[:10]:
        vals = [f"{r:.3f}" for r in rewards]
        print(f"{sid:<45}" + "".join(f"{v:>12}" for v in vals))

    print()


def section_errors(all_models):
    print("=" * 90)
    print("  6. ERROR & EDGE CASE ANALYSIS")
    print("=" * 90)
    print()

    # Errors per model
    for model, data, traces in all_models:
        errors = [t for t in traces if t.get("error") is not None]
        if errors:
            print(f"{_short_model(model)} — {len(errors)} errors:")
            for t in errors:
                sid = t.get("scenario_id", "?")
                err = t.get("error", "")[:80]
                print(f"  {sid:<45} {err}")
            print()

    # No-vulnerability (false positive analysis)
    print("No-Vulnerability Class (false positive detection):")
    labels = [_short_model(m) for m, _, _ in all_models]
    header = f"{'Metric':<30}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    for model, _, traces in all_models:
        novuln = [t for t in traces if t.get("error") is None
                  and t.get("info", {}).get("canonical_category") == "no-vulnerability"]
        if novuln:
            # How many correctly identified as no-vuln?
            correct = sum(1 for t in novuln if t["subscores"].get("category_match", 0) >= 0.9)
            break
    # Print in table format
    vals_mean = []
    vals_correct = []
    vals_total = []
    for model, _, traces in all_models:
        novuln = [t for t in traces if t.get("error") is None
                  and t.get("info", {}).get("canonical_category") == "no-vulnerability"]
        correct = sum(1 for t in novuln if t.get("subscores", {}).get("category_match", 0) >= 0.9)
        vals_mean.append(f"{_mean([t['reward'] for t in novuln]):.3f}" if novuln else "—")
        vals_correct.append(f"{correct}/{len(novuln)}" if novuln else "—")
        vals_total.append(str(len(novuln)))

    print(f"{'Count':<30}" + "".join(f"{v:>15}" for v in vals_total))
    print(f"{'Mean Reward':<30}" + "".join(f"{v:>15}" for v in vals_mean))
    print(f"{'Correctly ID as safe':<30}" + "".join(f"{v:>15}" for v in vals_correct))

    # Hint usage
    print()
    print("Hint Usage (triggers 0.7× reward penalty):")
    header = f"{'Metric':<30}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    for stat_name, fn in [
        ("Traces w/ hints used", lambda t: t.get("info", {}).get("hints_used", False)),
        ("Hint usage rate", None),
    ]:
        vals = []
        for model, _, traces in all_models:
            good = [t for t in traces if t.get("error") is None]
            if stat_name == "Hint usage rate":
                hinted = sum(1 for t in good if t.get("info", {}).get("hints_used", False))
                vals.append(_pct(hinted, len(good)))
            else:
                count = sum(1 for t in good if fn(t))
                vals.append(str(count))
        print(f"{stat_name:<30}" + "".join(f"{v:>15}" for v in vals))

    print()


def section_tool_usage(all_models):
    print("=" * 90)
    print("  7. TOOL USAGE & INTERACTION PATTERNS")
    print("=" * 90)
    print()

    labels = [_short_model(m) for m, _, _ in all_models]
    difficulties = ["easy", "medium", "hard"]

    # Steps
    print("Average Steps (interaction turns per episode):")
    header = f"{'Metric':<22}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))

    # Overall
    vals = []
    for model, _, traces in all_models:
        good = [t for t in traces if t.get("error") is None]
        steps = [t.get("steps", 0) for t in good]
        vals.append(f"{_mean(steps):.2f}")
    print(f"{'Overall':<22}" + "".join(f"{v:>15}" for v in vals))

    # By difficulty
    for diff in difficulties:
        vals = []
        for model, _, traces in all_models:
            dt = [t for t in traces if t.get("error") is None
                  and t.get("info", {}).get("difficulty") == diff]
            steps = [t.get("steps", 0) for t in dt]
            vals.append(f"{_mean(steps):.2f}")
        print(f"{'  ' + diff:<22}" + "".join(f"{v:>15}" for v in vals))

    # Code read rate
    print()
    print("Code Read Rate (% of episodes where read_code() was called):")
    vals = []
    for model, _, traces in all_models:
        good = [t for t in traces if t.get("error") is None]
        read = sum(1 for t in good if t.get("info", {}).get("code_read", False))
        vals.append(_pct(read, len(good)))
    header = f"{'Metric':<22}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    print(f"{'Code read rate':<22}" + "".join(f"{v:>15}" for v in vals))

    # Schema penalty
    print()
    print("Schema Penalty Distribution:")
    vals_full = []
    vals_penalized = []
    for model, _, traces in all_models:
        good = [t for t in traces if t.get("error") is None]
        full = sum(1 for t in good if t.get("info", {}).get("schema_penalty", 1.0) == 1.0)
        vals_full.append(_pct(full, len(good)))
        penalized = sum(1 for t in good if t.get("info", {}).get("schema_penalty", 1.0) < 1.0)
        vals_penalized.append(_pct(penalized, len(good)))
    header = f"{'Metric':<22}" + "".join(f"{l:>15}" for l in labels)
    print(header)
    print("-" * len(header))
    print(f"{'No penalty (1.0)':<22}" + "".join(f"{v:>15}" for v in vals_full))
    print(f"{'Penalized (<1.0)':<22}" + "".join(f"{v:>15}" for v in vals_penalized))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep-dive model comparison")
    parser.add_argument("--results", nargs="+", default=None,
                        help="Result JSON files to compare (auto-detects canonical runs if omitted)")
    parser.add_argument("--brief", action="store_true",
                        help="Print only overall summary and difficulty breakdown")
    args = parser.parse_args()

    if args.results:
        result_files = args.results
    else:
        # Auto-detect canonical runs (largest file per model)
        results_dir = Path("results")
        if not results_dir.exists():
            print("No results/ directory found. Run evaluations first.")
            return

        # Group by model, pick largest file (full run)
        by_model = defaultdict(list)
        for f in results_dir.glob("*.json"):
            by_model[f.stem.rsplit("_", 2)[0]].append(f)

        result_files = []
        for model, files in sorted(by_model.items()):
            biggest = max(files, key=lambda f: f.stat().st_size)
            result_files.append(str(biggest))

    if not result_files:
        print("No result files found.")
        return

    all_models = load_results(result_files)
    scenario_index = build_scenario_index(all_models)

    model_names = ", ".join(_short_model(m) for m, _, _ in all_models)
    print()
    print(f"╔{'═' * 88}╗")
    print(f"║  SOLIDITY VULNERABILITY DETECTION — MODEL COMPARISON{' ' * 35}║")
    print(f"║  Models: {model_names:<77}║")
    print(f"╚{'═' * 88}╝")
    print()

    section_overall(all_models)

    if args.brief:
        section_difficulty(all_models)
        return

    section_subscores(all_models)
    section_difficulty(all_models)
    section_categories(all_models)
    section_head_to_head(all_models, scenario_index)
    section_errors(all_models)
    section_tool_usage(all_models)

    print("=" * 90)
    print("  END OF REPORT")
    print("=" * 90)


if __name__ == "__main__":
    main()
