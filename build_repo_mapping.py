"""Build scenario -> repo mapping for agentic eval.

Scans data/repos/ to find which repo each scenario's source_file belongs to.
Writes data/repo_mapping.json: { scenario_id: "org/repo", ... }

Usage:
    python build_repo_mapping.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def build_sol_index(repos_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Build reverse index: relative .sol path -> [(repo_key, full_path), ...]."""
    index: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for org_dir in sorted(repos_dir.iterdir()):
        if not org_dir.is_dir() or org_dir.name.startswith("."):
            continue
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith("."):
                continue
            repo_key = f"{org_dir.name}/{repo_dir.name}"
            for sol_file in repo_dir.rglob("*.sol"):
                # Skip node_modules, .git, lib (dependencies)
                parts = sol_file.relative_to(repo_dir).parts
                if any(p in (".git", "node_modules") for p in parts):
                    continue
                rel_path = str(sol_file.relative_to(repo_dir))
                index[rel_path].append((repo_key, sol_file))
    return index


def disambiguate_by_content(
    candidates: list[tuple[str, Path]], code_clean: str
) -> str | None:
    """Pick the repo whose file content best matches the scenario's code_clean."""
    if not code_clean:
        return candidates[0][0] if candidates else None

    # Compare first 500 chars of code_clean against each candidate file
    snippet = code_clean[:500]
    best_repo = None
    best_overlap = 0
    for repo_key, full_path in candidates:
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Simple overlap: count matching lines
        snippet_lines = set(snippet.split("\n"))
        content_lines = set(content.split("\n"))
        overlap = len(snippet_lines & content_lines)
        if overlap > best_overlap:
            best_overlap = overlap
            best_repo = repo_key
    return best_repo


def build_mapping(scenarios_path: Path, repos_dir: Path) -> dict[str, str]:
    """Build {scenario_id: repo_key} mapping."""
    with open(scenarios_path) as f:
        scenarios = json.load(f)

    print(f"Loaded {len(scenarios)} scenarios")

    # Filter to scenarios with source_file
    with_source = [s for s in scenarios if s.get("source_file")]
    print(f"Scenarios with source_file: {len(with_source)}")

    # Build .sol file index
    print("Indexing .sol files in repos...")
    sol_index = build_sol_index(repos_dir)
    print(f"Indexed {sum(len(v) for v in sol_index.values())} files across {len(sol_index)} unique paths")

    # Map scenarios to repos
    mapping: dict[str, str] = {}
    stats = {"unique": 0, "disambiguated": 0, "not_found": 0}

    for s in with_source:
        source_file = s["source_file"]
        candidates = sol_index.get(source_file, [])

        if len(candidates) == 1:
            mapping[s["id"]] = candidates[0][0]
            stats["unique"] += 1
        elif len(candidates) > 1:
            repo_key = disambiguate_by_content(candidates, s.get("code_clean", ""))
            if repo_key:
                mapping[s["id"]] = repo_key
                stats["disambiguated"] += 1
            else:
                stats["not_found"] += 1
        else:
            stats["not_found"] += 1

    print(f"\nMapping results:")
    print(f"  Unique match: {stats['unique']}")
    print(f"  Disambiguated: {stats['disambiguated']}")
    print(f"  Not found: {stats['not_found']}")
    print(f"  Total mapped: {len(mapping)}")

    return mapping


def main():
    data_dir = Path(__file__).parent / "data"
    scenarios_path = data_dir / "scenarios.json"
    repos_dir = data_dir / "repos"

    if not scenarios_path.exists():
        print(f"Error: {scenarios_path} not found")
        sys.exit(1)
    if not repos_dir.exists():
        print(f"Error: {repos_dir} not found")
        sys.exit(1)

    mapping = build_mapping(scenarios_path, repos_dir)

    out_path = data_dir / "repo_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nWrote {out_path} ({len(mapping)} entries)")


if __name__ == "__main__":
    main()
