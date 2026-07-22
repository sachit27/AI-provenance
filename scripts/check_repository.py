#!/usr/bin/env python3
"""Credential-free structural checks for the public repository."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_input_hashes() -> None:
    manifest = ROOT / "results" / "input_hashes.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"Missing input listed in hash manifest: {relative}")
        observed = sha256(path)
        if observed != expected:
            raise AssertionError(f"SHA-256 mismatch for {relative}")


def check_reference_results() -> None:
    path = ROOT / "results" / "crossfit_confirmatory_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = payload["primary_family"]["tests"]
    rows = payload["rows"]
    if len(tests) != 8:
        raise AssertionError(f"Expected 8 primary tests, found {len(tests)}")
    if len(rows) != 4:
        raise AssertionError(f"Expected 4 topic-model rows, found {len(rows)}")

    expected_pairs = {
        ("education", "openai"),
        ("education", "mpnet"),
        ("trust", "openai"),
        ("trust", "mpnet"),
    }
    observed_pairs = {(row["topic"], row["model"]) for row in rows}
    if observed_pairs != expected_pairs:
        raise AssertionError(f"Unexpected topic-model rows: {observed_pairs}")

    numeric_fields = (
        "official_mean_coverage",
        "mean_optimized_difference",
        "mean_optimized_bootstrap_ci_low",
        "mean_optimized_bootstrap_ci_high",
        "official_bottom_decile_mean",
        "tail_optimized_bottom_difference",
        "tail_optimized_bootstrap_ci_low",
        "tail_optimized_bootstrap_ci_high",
    )
    for row in rows:
        for field in numeric_fields:
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise AssertionError(f"Invalid {field} in {row['topic']}/{row['model']}")
        if not (
            row["mean_optimized_bootstrap_ci_low"]
            <= row["mean_optimized_difference"]
            <= row["mean_optimized_bootstrap_ci_high"]
        ):
            raise AssertionError("Mean-coverage estimate lies outside its interval")
        if not (
            row["tail_optimized_bootstrap_ci_low"]
            <= row["tail_optimized_bottom_difference"]
            <= row["tail_optimized_bootstrap_ci_high"]
        ):
            raise AssertionError("Bottom-decile estimate lies outside its interval")


def check_python_syntax() -> None:
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in sorted(directory.glob("*.py")):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def check_tracked_files() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked = [item for item in result.stdout.decode().split("\0") if item]
    forbidden = [
        path
        for path in tracked
        if Path(path).name == ".env"
        or path.endswith(".pyc")
        or "__pycache__" in Path(path).parts
    ]
    if forbidden:
        raise AssertionError(f"Generated or credential files are tracked: {forbidden}")


def main() -> None:
    check_input_hashes()
    check_reference_results()
    check_python_syntax()
    check_tracked_files()
    print("PASS: repository structure, hashes, results, and Python syntax verified")


if __name__ == "__main__":
    main()
