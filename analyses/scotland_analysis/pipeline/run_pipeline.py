#!/usr/bin/env python3
"""Run the (unchanged) analysis in its documented order on the Scottish inputs.

Port of the original ``pipeline/run_pipeline.py``.  Only differences: the
Scottish input builder is run first, the Canadian-submission guards/audits/figure
stage are replaced by the Scottish summary/export scripts, and ``--stage
preprocess`` / ``--skip-preprocess`` allow stage 01 to be run on its own.

Typical use (from the canadian_data root):

    analyses/.venv/bin/python analyses/scotland_analysis/pipeline/run_pipeline.py \
        --from-raw --output-root analyses/scotland_analysis/analysis
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent            # .../analyses/scotland_analysis/pipeline
ROOT = HERE.parent                                # .../analyses/scotland_analysis
BASE = HERE.parents[2]                            # canadian_data


def run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="regenerate analysis-ready records and embeddings before analysis",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="analysis directory (default: analyses/scotland_analysis/analysis)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess"],
        default="all",
        help="'preprocess' builds the inputs and runs stage 01 only (writes "
             "<q>/clean.parquet and the nomic embeddings), then stops",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="with --from-raw: skip stage 01 (clean.parquet must already exist)",
    )
    args = parser.parse_args()

    canadian_submitted = (BASE / "src" / "analysis").resolve()
    default_output = ROOT / "analysis"
    output_root = (args.output_root or default_output).expanduser().resolve()
    if output_root == canadian_submitted:
        parser.error("the Scottish run must never write to the Canadian pipeline/analysis")
    os.environ["PROVENANCE_OUTPUT_DIR"] = str(output_root)
    os.environ["PROVENANCE_FIGURE_DIR"] = str(output_root / "figures")

    from_raw = args.from_raw or args.stage == "preprocess"
    if args.skip_preprocess and args.stage == "preprocess":
        parser.error("--skip-preprocess and --stage preprocess are contradictory")
    if args.skip_preprocess and not from_raw:
        parser.error("--skip-preprocess only applies with --from-raw")

    if from_raw:
        run("build_inputs.py")
        if args.skip_preprocess:
            missing = [q for q in ("q4", "q7", "q20", "q21", "q22", "q23")
                       if not (output_root / q / "clean.parquet").exists()]
            if missing:
                parser.error(f"--skip-preprocess but clean.parquet is missing for {missing}")
        else:
            run("pipeline/01_preprocess.py")
        if args.stage == "preprocess":
            print(f"\nStage 01 complete: {output_root}/<q>/clean.parquet")
            return
        for model in ("openai", "mpnet"):
            run(
                "pipeline/02_prepare_benchmarks.py",
                "--model", model,
                "--prepare-only",
            )

    for script in (
        "pipeline/03_topology.py",
        "pipeline/04_transport.py",
        "pipeline/05_associations.py",
        "pipeline/06_cross_topic.py",
    ):
        run(script)

    for model in ("openai", "mpnet"):
        run("pipeline/02_prepare_benchmarks.py", "--model", model)
    for model in ("openai", "mpnet"):
        run("pipeline/07_crossfit_benchmarks.py", "--model", model)

    run("pipeline/08_summarize_benchmarks.py")
    # Stage 09 (Canadian submission figures) and the pipeline/analysis hash
    # audits are intentionally not ported; see DIFF_FROM_ORIGINAL.md.
    run("export_coverage.py", "--analysis-root", str(output_root))
    run("summarize_scotland.py", "--analysis-root", str(output_root))


if __name__ == "__main__":
    main()
