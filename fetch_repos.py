"""Scan findings for GitHub source URLs and clone referenced contest repos.

Usage:
    python fetch_repos.py /path/to/protocol-vulnerabilities-index

Outputs:
    data/repos/{org}/{repo}/   — Full clones of referenced repos
    data/repos/manifest.json   — Clone status for each repo
"""

import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Matches GitHub URLs pointing to .sol files with optional line ranges
# e.g. https://github.com/code-423n4/2022-12-tigris/blob/main/contracts/TradingLibrary.sol#L91-L122
GITHUB_SOL_URL_RE = re.compile(
    r"https://github\.com/"
    r"(?P<org>[^/]+)/(?P<repo>[^/]+)"
    r"/blob/[^/]+/"
    r"(?P<filepath>[^\s)>\]#]+\.sol)"
    r"(?:#L(?P<start>\d+)(?:-L(?P<end>\d+))?)?"
)


def scan_findings_for_repos(index_repo: Path) -> dict[str, set[str]]:
    """Scan all findings JSON files and collect unique repo → set of referenced filepaths."""
    findings_dir = index_repo / "data" / "findings" / "protocols"
    repo_files: dict[str, set[str]] = defaultdict(set)

    for json_file in sorted(findings_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for finding in data:
            content = finding.get("content", "")
            for m in GITHUB_SOL_URL_RE.finditer(content):
                repo_key = f"{m.group('org')}/{m.group('repo')}"
                repo_files[repo_key].add(m.group("filepath"))

    return dict(repo_files)


def clone_repo(repo_key: str, dest: Path) -> bool:
    """Clone a repo via gh (full clone). Returns True on success."""
    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["gh", "repo", "clone", repo_key, str(dest)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True
        # Common failure: repo not found, private, etc.
        print(f"  FAILED ({result.returncode}): {result.stderr.strip()[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {repo_key}")
        return False
    except FileNotFoundError:
        print("ERROR: 'gh' CLI not found. Install it: https://cli.github.com/")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_repos.py /path/to/protocol-vulnerabilities-index")
        sys.exit(1)

    index_repo = Path(sys.argv[1])
    findings_dir = index_repo / "data" / "findings" / "protocols"
    if not findings_dir.exists():
        print(f"Error: {findings_dir} not found")
        sys.exit(1)

    repos_dir = Path(__file__).parent / "data" / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Scan for referenced repos
    print("Scanning findings for GitHub source URLs...")
    repo_files = scan_findings_for_repos(index_repo)
    print(f"Found {len(repo_files)} unique repos referenced across all findings")

    # Show top repos by file count
    sorted_repos = sorted(repo_files.items(), key=lambda x: -len(x[1]))
    print(f"\nTop 20 repos by referenced file count:")
    for repo_key, files in sorted_repos[:20]:
        print(f"  {repo_key}: {len(files)} files")

    # Clone repos
    manifest = {}
    success_count = 0
    fail_count = 0

    for i, (repo_key, files) in enumerate(sorted_repos, 1):
        dest = repos_dir / repo_key
        print(f"\n[{i}/{len(sorted_repos)}] {repo_key} ({len(files)} files)...", end=" ")

        if dest.exists():
            print("already cloned")
            manifest[repo_key] = {"status": "ok", "files_referenced": len(files)}
            success_count += 1
            continue

        ok = clone_repo(repo_key, dest)
        if ok:
            print("OK")
            manifest[repo_key] = {"status": "ok", "files_referenced": len(files)}
            success_count += 1
        else:
            manifest[repo_key] = {"status": "failed", "files_referenced": len(files)}
            fail_count += 1

        # Small delay between clones to be respectful
        time.sleep(0.5)

    # Write manifest
    manifest_path = repos_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {success_count} cloned, {fail_count} failed, {len(sorted_repos)} total")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
