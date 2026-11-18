#!/usr/bin/env python3
"""Collect the Scottish original-pipeline results into RESULTS.md + CSV.

Reads only stage outputs under ``--analysis-root`` (default
``revision/scotland_original/analysis``):

* ``<q>/metadata.json``                 stage 01 retention counts
* ``<q>/topology.json``                 number of regions (K-means k)
* ``<q>/participant_coverage.csv``      per-participant coverage, all 3 models
                                        (created by export_coverage.py if absent)
* ``crossfit_confirmatory_summary.csv`` stage 08 (openai, mpnet): exact-length
                                        random-window p-values, cross-fitted held-out
                                        gains, 95% bootstrap intervals, raw and Holm p

Writes ``revision/scotland_original/RESULTS.md`` and
``revision/scotland_original/scotland_original_summary.csv``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_coverage as ec  # noqa: E402

TOPIC_ORDER = ["q4", "q7", "q20", "q21", "q22", "q23"]
MODEL_ORDER = ["openai", "mpnet", "nomic"]
MODEL_NAMES = {"openai": "text-embedding-3-large", "mpnet": "all-mpnet-base-v2",
               "nomic": "nomic-embed-text (robustness)"}


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n < 2 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * values.sum()))


def bottom_decile_mean(values: np.ndarray) -> float:
    tail_n = max(1, int(math.ceil(0.10 * len(values))))
    return float(np.partition(values, tail_n - 1)[:tail_n].mean())


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def markdown_table(frame: pd.DataFrame, digits: dict | None = None) -> str:
    digits = digits or {}
    lines = ["| " + " | ".join(frame.columns) + " |",
             "|" + "|".join("---" for _ in frame.columns) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[c], digits.get(c, 3)) if not isinstance(row[c], str)
                                        else row[c] for c in frame.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, default=HERE / "analysis")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    root = args.analysis_root.expanduser().resolve()
    topics = [t for t in TOPIC_ORDER if (root / t / "clustered.parquet").exists()]
    if not topics:
        raise SystemExit(f"no completed topics under {root}")

    confirm_path = root / "crossfit_confirmatory_summary.csv"
    confirm = pd.read_csv(confirm_path) if confirm_path.exists() else pd.DataFrame()
    confirm_json = root / "crossfit_confirmatory_summary.json"
    family_size = None
    if confirm_json.exists():
        family_size = len(json.loads(confirm_json.read_text())["primary_family"]["tests"])

    rows, retention = [], []
    for topic in topics:
        base = root / topic
        meta = json.loads((base / "metadata.json").read_text())
        topology = json.loads((base / "topology.json").read_text())
        transport = json.loads((base / "transport.json").read_text())
        retention.append({
            "topic": topic, "label": meta["label"], "raw rows": meta["n_raw"],
            "empty": meta["n_empty"], "French heuristic": meta["n_french"],
            "< 5 words": meta["n_short"], "spam": meta["n_spam"],
            "off-topic (relevance filter)": meta["n_off_topic"], "retained N": meta["n_final"],
            "official sentences": meta["n_summary_sentences"],
            "official words": sum(len(s["sentence"].split())
                                  for s in transport["sentence_mass"].values()),
        })
        csv_path = base / "participant_coverage.csv"
        if not csv_path.exists():
            ec.export_topic(base, base, topic, ec.MODELS)
        coverage = pd.read_csv(csv_path)
        for model in MODEL_ORDER:
            sub = coverage[coverage["embedding_model"] == model]
            if sub.empty:
                continue
            values = sub["coverage"].to_numpy(dtype=float)
            row = {
                "topic": topic, "label": meta["label"], "model": model,
                "embedding_model": MODEL_NAMES[model],
                "n_retained": len(sub),
                "n_regions": int(topology["kmeans"]["best_k"]),
                "official_mean_coverage": float(values.mean()),
                "bottom_decile_mean": bottom_decile_mean(values),
                "low_coverage_rate": float(sub["low_coverage"].mean()),
                "gini": gini(values),
            }
            if not confirm.empty:
                match = confirm[(confirm["topic"] == topic) & (confirm["model"] == model)]
                if len(match) == 1:
                    m = match.iloc[0]
                    row.update({
                        "official_mean_coverage": float(m["official_mean_coverage"]),
                        "bottom_decile_mean": float(m["official_bottom_decile_mean"]),
                        "low_coverage_rate": float(m["official_exclusion_rate"]),
                        "exact_random_null_mean": float(m["exact_random_null_mean"]),
                        "exact_random_mean_p": float(m["exact_random_mean_p"]),
                        "exact_random_null_bottom_decile": float(m["exact_random_null_bottom_decile"]),
                        "exact_random_bottom_decile_p": float(m["exact_random_bottom_decile_p"]),
                        "mean_opt_heldout_coverage": float(m["mean_optimized_coverage"]),
                        "mean_opt_gain": float(m["mean_optimized_difference"]),
                        "mean_opt_gain_ci_low": float(m["mean_optimized_bootstrap_ci_low"]),
                        "mean_opt_gain_ci_high": float(m["mean_optimized_bootstrap_ci_high"]),
                        "mean_opt_p_raw": float(m["mean_optimized_p_raw"]),
                        "mean_opt_p_holm": float(m["mean_optimized_p_holm_primary_family"]),
                        "tail_opt_heldout_bottom_decile": float(m["tail_optimized_bottom_decile_mean"]),
                        "tail_opt_gain": float(m["tail_optimized_bottom_difference"]),
                        "tail_opt_gain_ci_low": float(m["tail_optimized_bootstrap_ci_low"]),
                        "tail_opt_gain_ci_high": float(m["tail_optimized_bootstrap_ci_high"]),
                        "tail_opt_p_raw": float(m["tail_optimized_p_raw"]),
                        "tail_opt_p_holm": float(m["tail_optimized_p_holm_primary_family"]),
                    })
            rows.append(row)

    frame = pd.DataFrame(rows)
    csv_out = args.out_dir / "scotland_original_summary.csv"
    frame.to_csv(csv_out, index=False)

    def view(columns, rename):
        sub = frame[["topic", "model"] + columns].rename(columns=rename)
        return sub

    parts = ["# Scotland: original pipeline (unchanged method, swapped inputs)", "",
             "Generated by `summarize_scotland.py` from the stage outputs in "
             f"`{root.name}/`. The analysis pipeline is the Canadian one (see "
             "`DIFF_FROM_ORIGINAL.md`); only the inputs changed. Coverage = each "
             "participant's maximum cosine similarity to an official-report sentence; "
             "low coverage = coverage below mean minus one SD (per model). Official "
             "targets are the report section(s) for each question, so budgets "
             "(sentences/words) differ by topic.", ""]
    parts += ["## Retention (stage 01)", "", markdown_table(pd.DataFrame(retention), {}), ""]
    parts += ["## Official coverage", "",
              markdown_table(view(
                  ["n_retained", "n_regions", "official_mean_coverage", "bottom_decile_mean",
                   "low_coverage_rate", "gini"],
                  {"n_retained": "N", "n_regions": "regions", "official_mean_coverage": "mean cov.",
                   "bottom_decile_mean": "bottom-decile mean", "low_coverage_rate": "low-cov. rate",
                   "gini": "Gini"}), {"N": 0, "regions": 0}), ""]
    if "exact_random_mean_p" in frame:
        sub = frame.dropna(subset=["exact_random_mean_p"])
        parts += ["## Official vs exact-length random windows (1,000 draws)", "",
                  "p = (1 + #{random >= official}) / 1001; small p means the official "
                  "summary beats chance. Stage 02 output.", "",
                  markdown_table(sub[["topic", "model", "official_mean_coverage",
                                      "exact_random_null_mean", "exact_random_mean_p",
                                      "bottom_decile_mean", "exact_random_null_bottom_decile",
                                      "exact_random_bottom_decile_p"]].rename(columns={
                                          "official_mean_coverage": "official mean",
                                          "exact_random_null_mean": "random mean",
                                          "exact_random_mean_p": "p (mean)",
                                          "bottom_decile_mean": "official bottom-decile",
                                          "exact_random_null_bottom_decile": "random bottom-decile",
                                          "exact_random_bottom_decile_p": "p (bottom-decile)"}),
                                {"p (mean)": 4, "p (bottom-decile)": 4}), ""]
        sub = frame.dropna(subset=["mean_opt_gain"])
        fam = f"{family_size}" if family_size else "all"
        parts += ["## Cross-fitted held-out gains over the official summary (5x5-fold)", "",
                  "Gain = held-out, budget-matched extractive summary minus official. "
                  "Mean-oriented: mean-optimized selection, metric = mean coverage. "
                  "Tail-oriented: balanced-tail selection, metric = bottom-decile mean. "
                  "Intervals are 95% participant-bootstrap percentile intervals; p is the "
                  f"one-sided paired-randomization p; Holm adjusts across the {fam} primary tests "
                  "(topics x models x two objectives).", "",
                  markdown_table(sub[["topic", "model", "mean_opt_gain", "mean_opt_gain_ci_low",
                                      "mean_opt_gain_ci_high", "mean_opt_p_raw", "mean_opt_p_holm"]].rename(
                      columns={"mean_opt_gain": "mean gain", "mean_opt_gain_ci_low": "CI low",
                               "mean_opt_gain_ci_high": "CI high", "mean_opt_p_raw": "p raw",
                               "mean_opt_p_holm": "p Holm"}),
                      {"p raw": 4, "p Holm": 4}), "",
                  markdown_table(sub[["topic", "model", "tail_opt_gain", "tail_opt_gain_ci_low",
                                      "tail_opt_gain_ci_high", "tail_opt_p_raw", "tail_opt_p_holm"]].rename(
                      columns={"tail_opt_gain": "tail gain", "tail_opt_gain_ci_low": "CI low",
                               "tail_opt_gain_ci_high": "CI high", "tail_opt_p_raw": "p raw",
                               "tail_opt_p_holm": "p Holm"}),
                      {"p raw": 4, "p Holm": 4}), ""]
    parts += ["## Notes", "",
              "- nomic is a robustness model in the original pipeline: it has coverage "
              "statistics only (no random-window or cross-fitted benchmark), shown as dashes.",
              "- Regions = K-means k chosen by the original multi-metric consensus with the "
              "stability override, on the OpenAI embeddings; the same partition is used for all models.",
              "- Per-participant values: `analysis/<q>/participant_coverage.csv` (export_coverage.py).",
              f"- Full numeric table: `{csv_out.name}`.", ""]
    (args.out_dir / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out_dir / 'RESULTS.md'} and {csv_out}")


if __name__ == "__main__":
    main()
