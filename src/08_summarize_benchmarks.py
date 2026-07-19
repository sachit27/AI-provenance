#!/usr/bin/env python3
"""Create a multiplicity-adjusted summary of cross-fitted benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR


TOPICS = ["education", "trust"]
MODELS = ["openai", "mpnet"]
PRIMARY_TESTS = [
    ("mean_optimized_extractive", "mean_coverage"),
    ("balanced_tail_optimized_extractive", "bottom_decile_mean"),
]


def holm_adjust(values: list[float]) -> list[float]:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--snapshot-root", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    payloads = {}
    cluster_labels = {}
    for topic in TOPICS:
        with open(args.snapshot_root / topic / "topology.json") as handle:
            cluster_labels[topic] = json.load(handle)["topic_labels"]
        for model in MODELS:
            path = args.input_root / topic / f"crossfit_{model}.json"
            with open(path) as handle:
                payloads[(topic, model)] = json.load(handle)

    tests = []
    for topic in TOPICS:
        for model in MODELS:
            comparisons = payloads[(topic, model)][
                "participant_level_bootstrap"
            ]["paired_vs_official"]
            for method, metric in PRIMARY_TESTS:
                tests.append({
                    "topic": topic,
                    "model": model,
                    "method": method,
                    "metric": metric,
                    "p_raw": comparisons[method][metric]["one_sided_improvement_p"],
                })
    adjusted = holm_adjust([test["p_raw"] for test in tests])
    for test, value in zip(tests, adjusted):
        test["p_holm_across_eight_primary_tests"] = float(value)
    adjusted_lookup = {
        (test["topic"], test["model"], test["method"], test["metric"]):
        test["p_holm_across_eight_primary_tests"]
        for test in tests
    }

    rows = []
    for topic in TOPICS:
        for model in MODELS:
            data = payloads[(topic, model)]
            bootstrap = data["participant_level_bootstrap"]
            points = bootstrap["point_metrics"]
            comparisons = bootstrap["paired_vs_official"]
            repeats = data["repeat_level_sensitivity"]["paired_vs_official"]
            groups = data["group_bootstrap_diagnostics"]
            worst_group = min(
                groups,
                key=lambda label: groups[label]["official_mean_coverage"],
            )
            worst = groups[worst_group]
            worst_tail = worst["methods"]["balanced_tail_optimized_extractive"]
            mean_result = comparisons["mean_optimized_extractive"]["mean_coverage"]
            tail_result = comparisons[
                "balanced_tail_optimized_extractive"
            ]["bottom_decile_mean"]
            exclusion_result = comparisons[
                "balanced_tail_optimized_extractive"
            ]["exclusion_rate_at_frozen_threshold"]
            exact = data["exact_length_random_reference"]
            rows.append({
                "topic": topic,
                "model": model,
                "n_participants": data["frozen_input_n"],
                "official_word_budget": data["official_total_words"],
                "official_mean_coverage": points["official"]["mean_coverage"],
                "mean_optimized_coverage": points["mean_optimized_extractive"]["mean_coverage"],
                "mean_optimized_difference": mean_result["point_difference"],
                "mean_optimized_bootstrap_ci_low": mean_result["percentile_95_ci"][0],
                "mean_optimized_bootstrap_ci_high": mean_result["percentile_95_ci"][1],
                "mean_optimized_repeat_mean_difference": repeats["mean_optimized_extractive"]["mean_coverage"]["mean_difference"],
                "mean_optimized_repeat_range_low": repeats["mean_optimized_extractive"]["mean_coverage"]["observed_range"][0],
                "mean_optimized_repeat_range_high": repeats["mean_optimized_extractive"]["mean_coverage"]["observed_range"][1],
                "mean_optimized_p_raw": mean_result["one_sided_improvement_p"],
                "mean_optimized_p_holm_primary_family": adjusted_lookup[(
                    topic, model, "mean_optimized_extractive", "mean_coverage"
                )],
                "official_bottom_decile_mean": points["official"]["bottom_decile_mean"],
                "tail_optimized_bottom_decile_mean": points["balanced_tail_optimized_extractive"]["bottom_decile_mean"],
                "tail_optimized_bottom_difference": tail_result["point_difference"],
                "tail_optimized_bootstrap_ci_low": tail_result["percentile_95_ci"][0],
                "tail_optimized_bootstrap_ci_high": tail_result["percentile_95_ci"][1],
                "tail_optimized_repeat_mean_difference": repeats["balanced_tail_optimized_extractive"]["bottom_decile_mean"]["mean_difference"],
                "tail_optimized_repeat_range_low": repeats["balanced_tail_optimized_extractive"]["bottom_decile_mean"]["observed_range"][0],
                "tail_optimized_repeat_range_high": repeats["balanced_tail_optimized_extractive"]["bottom_decile_mean"]["observed_range"][1],
                "tail_optimized_p_raw": tail_result["one_sided_improvement_p"],
                "tail_optimized_p_holm_primary_family": adjusted_lookup[(
                    topic, model, "balanced_tail_optimized_extractive", "bottom_decile_mean"
                )],
                "official_exclusion_rate": points["official"]["exclusion_rate_at_frozen_threshold"],
                "tail_optimized_exclusion_rate": points["balanced_tail_optimized_extractive"]["exclusion_rate_at_frozen_threshold"],
                "tail_optimized_exclusion_difference": exclusion_result["point_difference"],
                "tail_optimized_exclusion_ci_low": exclusion_result["percentile_95_ci"][0],
                "tail_optimized_exclusion_ci_high": exclusion_result["percentile_95_ci"][1],
                "worst_cluster_id": int(worst_group),
                "worst_cluster_label": cluster_labels[topic][worst_group],
                "worst_cluster_n": worst["n"],
                "worst_cluster_official_mean": worst["official_mean_coverage"],
                "worst_cluster_tail_optimized_mean": worst_tail["mean_coverage"],
                "worst_cluster_mean_difference_ci_low": worst_tail["mean_difference_percentile_95_ci"][0],
                "worst_cluster_mean_difference_ci_high": worst_tail["mean_difference_percentile_95_ci"][1],
                "worst_cluster_official_exclusion": worst["official_exclusion_rate"],
                "worst_cluster_tail_optimized_exclusion": worst_tail["exclusion_rate"],
                "worst_cluster_exclusion_difference_ci_low": worst_tail["exclusion_difference_percentile_95_ci"][0],
                "worst_cluster_exclusion_difference_ci_high": worst_tail["exclusion_difference_percentile_95_ci"][1],
                "exact_random_official_mean": exact["official"]["mean_coverage"],
                "exact_random_null_mean": exact["random_mean_coverage"]["mean"],
                "exact_random_mean_p": exact["random_mean_coverage"]["official_upper_tail_p"],
                "exact_random_official_bottom_decile": exact["official"]["bottom_decile_mean"],
                "exact_random_null_bottom_decile": exact["random_bottom_decile_mean"]["mean"],
                "exact_random_bottom_decile_p": exact["random_bottom_decile_mean"]["official_upper_tail_p"],
            })

    frame = pd.DataFrame(rows)
    csv_path = args.input_root / "crossfit_confirmatory_summary.csv"
    json_path = args.input_root / "crossfit_confirmatory_summary.json"
    frame.to_csv(csv_path, index=False)
    with open(json_path, "w") as handle:
        json.dump({
            "primary_family": {
                "description": (
                    "Eight one-sided improvement tests: mean-optimized mean coverage "
                    "and tail-optimized bottom-decile coverage across two topics and two models."
                ),
                "adjustment": "Holm family-wise error control",
                "tests": tests,
            },
            "rows": rows,
        }, handle, indent=2)
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
