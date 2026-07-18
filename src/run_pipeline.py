#!/usr/bin/env python3
"""Run the participatory-provenance analysis from the two source CSV files."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(HERE / script), *arguments]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete analysis from data/ into a new output directory."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "analysis",
        help="directory for generated records, embeddings, results, and figures",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root in {ROOT.resolve(), (ROOT / "data").resolve(), HERE.resolve()}:
        parser.error("--output-root must be a dedicated output directory")
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["PROVENANCE_OUTPUT_DIR"] = str(output_root)
    os.environ["PROVENANCE_FIGURE_DIR"] = str(output_root / "figures")

    run("01_preprocess.py")
    for model in ("openai", "mpnet"):
        run("02_prepare_benchmarks.py", "--model", model, "--prepare-only")

    for script in (
        "03_topology.py",
        "04_transport.py",
        "05_associations.py",
        "06_cross_topic.py",
    ):
        run(script)

    for model in ("openai", "mpnet"):
        run("02_prepare_benchmarks.py", "--model", model)
    for model in ("openai", "mpnet"):
        run("07_crossfit_benchmarks.py", "--model", model)

    run("08_summarize_benchmarks.py")
    run("09_figures.py")
    print(f"\nAnalysis complete: {output_root}")


if __name__ == "__main__":
    main()
