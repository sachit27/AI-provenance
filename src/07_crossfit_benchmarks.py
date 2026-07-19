#!/usr/bin/env python3
"""Repeated cross-fitted refinement of the budget-matched benchmark.

Every participant is evaluated out of sample exactly once in each repetition
of shuffled five-fold cross-fitting. Fold assignment and candidate selection
do not use the full-corpus cluster partition. Participant-level out-of-fold
coverage is averaged across repetitions before conditional paired-bootstrap
intervals and paired randomization tests are computed. The script uses the
leakage-controlled candidate pool and combined embedding artifact created by
02_prepare_benchmarks.py.

The extractive selections remain coverage benchmarks, not publishable prose.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


SCRIPT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "budget_matched_benchmarks",
    SCRIPT_DIR / "02_prepare_benchmarks.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load benchmark helper module")
bm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bm)


SEED = 20260717
N_REPEATS = 5
N_FOLDS = 5
N_COMPLETE_RANDOM_PER_FOLD = 100
BOOTSTRAP_B = 5000
TAIL_FRACTION = 0.10

METHODS = [
    "official",
    "complete_sentence_random_expected",
    "mean_optimized_extractive",
    "balanced_tail_optimized_extractive",
]
# Participant-level inference is restricted to deterministic optimized-vs-
# official comparisons. Averaging many random summaries for each participant
# changes the lower-tail estimand; the summary-level chance null from script 14
# remains the valid random-baseline analysis.
COMPARISON_METHODS = METHODS[2:]
BOOTSTRAP_METRICS = [
    "mean_coverage",
    "bottom_decile_mean",
    "exclusion_rate_at_frozen_threshold",
    "gini",
]
LOWER_IS_BETTER = {
    "exclusion_rate_at_frozen_threshold",
    "gini",
}


def scalar_metrics(coverage: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(coverage, dtype=float)
    tail_n = max(1, int(math.ceil(TAIL_FRACTION * len(values))))
    tail = np.partition(values, tail_n - 1)[:tail_n]
    return {
        "mean_coverage": float(values.mean()),
        "bottom_decile_mean": float(tail.mean()),
        "exclusion_rate_at_frozen_threshold": float(np.mean(values < threshold)),
        "gini": bm.gini(values),
    }


def participant_bootstrap_indices(n: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.randint(0, n, size=n)


def bootstrap_interval(values: np.ndarray) -> list[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def paired_participant_bootstrap(
    predictions: dict[str, np.ndarray],
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    official = predictions["official"]
    point_metrics = {
        method: scalar_metrics(values, threshold)
        for method, values in predictions.items()
    }
    distributions = {
        method: {metric: np.empty(n_bootstrap, dtype=np.float32)
                 for metric in BOOTSTRAP_METRICS}
        for method in COMPARISON_METHODS
    }
    randomization_null = {
        method: {metric: np.empty(n_bootstrap, dtype=np.float32)
                 for metric in BOOTSTRAP_METRICS}
        for method in COMPARISON_METHODS
    }
    rng = np.random.RandomState(seed)
    for iteration in range(n_bootstrap):
        indices = participant_bootstrap_indices(len(official), rng)
        official_metrics = scalar_metrics(official[indices], threshold)
        for method in COMPARISON_METHODS:
            method_metrics = scalar_metrics(predictions[method][indices], threshold)
            for metric in BOOTSTRAP_METRICS:
                distributions[method][metric][iteration] = (
                    method_metrics[metric] - official_metrics[metric]
                )

    randomization_rng = np.random.RandomState(seed + 1)
    for method in COMPARISON_METHODS:
        method_values = predictions[method]
        for iteration in range(n_bootstrap):
            swap = randomization_rng.randint(0, 2, size=len(official)).astype(bool)
            permuted_official = np.where(swap, method_values, official)
            permuted_method = np.where(swap, official, method_values)
            official_metrics = scalar_metrics(permuted_official, threshold)
            method_metrics = scalar_metrics(permuted_method, threshold)
            for metric in BOOTSTRAP_METRICS:
                randomization_null[method][metric][iteration] = (
                    method_metrics[metric] - official_metrics[metric]
                )

    output = {
        "n_bootstrap": n_bootstrap,
        "n_paired_randomizations": n_bootstrap,
        "resampling_unit": "participant",
        "resampling_stratified_by_frozen_cluster": False,
        "predictions_averaged_across_crossfit_repetitions": True,
        "interval_estimand": (
            "Conditional on the fitted crossfit selections and generated embeddings"
        ),
        "hypothesis_test": (
            "Paired participant-label randomization; method and official scores are "
            "swapped within participants under the null"
        ),
        "point_metrics": point_metrics,
        "paired_vs_official": {},
    }
    for method in COMPARISON_METHODS:
        output["paired_vs_official"][method] = {}
        for metric in BOOTSTRAP_METRICS:
            values = distributions[method][metric]
            point_difference = (
                point_metrics[method][metric] - point_metrics["official"][metric]
            )
            null_values = randomization_null[method][metric]
            if metric in LOWER_IS_BETTER:
                one_sided = float(
                    (1 + np.sum(null_values <= point_difference)) /
                    (n_bootstrap + 1)
                )
            else:
                one_sided = float(
                    (1 + np.sum(null_values >= point_difference)) /
                    (n_bootstrap + 1)
                )
            two_sided = float(
                (1 + np.sum(np.abs(null_values) >= abs(point_difference))) /
                (n_bootstrap + 1)
            )
            output["paired_vs_official"][method][metric] = {
                "point_difference": float(point_difference),
                "percentile_95_ci": bootstrap_interval(values),
                "one_sided_improvement_p": one_sided,
                "two_sided_p": two_sided,
            }
    return output


def repeat_level_sensitivity(
    prediction_matrices: dict[str, np.ndarray],
    threshold: float,
) -> dict:
    n_repeats = prediction_matrices["official"].shape[0]
    output = {
        "n_complete_crossfit_repetitions": n_repeats,
        "summary": (
            "Observed variation across repeated partitions; repetitions reuse the "
            "same participants and are not independent population replications"
        ),
        "paired_vs_official": {},
    }
    for method in COMPARISON_METHODS:
        output["paired_vs_official"][method] = {}
        for metric in BOOTSTRAP_METRICS:
            differences = []
            for repeat_index in range(n_repeats):
                official_metrics = scalar_metrics(
                    prediction_matrices["official"][repeat_index], threshold
                )
                method_metrics = scalar_metrics(
                    prediction_matrices[method][repeat_index], threshold
                )
                differences.append(method_metrics[metric] - official_metrics[metric])
            values = np.asarray(differences, dtype=float)
            if metric in LOWER_IS_BETTER:
                all_improve = bool(np.all(values < 0))
            else:
                all_improve = bool(np.all(values > 0))
            output["paired_vs_official"][method][metric] = {
                "repeat_differences": values.tolist(),
                "mean_difference": float(values.mean()),
                "sd_across_repetitions": float(values.std(ddof=1)),
                "observed_range": [float(values.min()), float(values.max())],
                "all_repetitions_improve": all_improve,
            }
    return output


def group_bootstrap_diagnostics(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
    threshold: float,
    n_bootstrap: int,
    seed: int,
) -> dict:
    diagnostics = {}
    for label in sorted(np.unique(labels)):
        group_indices = np.flatnonzero(labels == label)
        official = predictions["official"][group_indices]
        group_result = {
            "n": int(len(group_indices)),
            "official_mean_coverage": float(official.mean()),
            "official_exclusion_rate": float(np.mean(official < threshold)),
            "methods": {},
        }
        rng = np.random.RandomState(seed + int(label) * 1009)
        resamples = rng.randint(
            0, len(group_indices), size=(n_bootstrap, len(group_indices))
        )
        official_resampled = official[resamples]
        official_means = official_resampled.mean(axis=1)
        official_exclusions = (official_resampled < threshold).mean(axis=1)
        for method in ["mean_optimized_extractive", "balanced_tail_optimized_extractive"]:
            values = predictions[method][group_indices]
            method_resampled = values[resamples]
            mean_differences = method_resampled.mean(axis=1) - official_means
            exclusion_differences = (
                (method_resampled < threshold).mean(axis=1) - official_exclusions
            )
            group_result["methods"][method] = {
                "mean_coverage": float(values.mean()),
                "mean_coverage_difference": float(values.mean() - official.mean()),
                "mean_difference_percentile_95_ci": bootstrap_interval(mean_differences),
                "exclusion_rate": float(np.mean(values < threshold)),
                "exclusion_rate_difference": float(
                    np.mean(values < threshold) - np.mean(official < threshold)
                ),
                "exclusion_difference_percentile_95_ci": bootstrap_interval(
                    exclusion_differences
                ),
            }
        diagnostics[str(int(label))] = group_result
    return diagnostics


def load_verified_embeddings(
    topic: str,
    model_name: str,
    snapshot_root: Path,
    benchmark_root: Path,
):
    topic_input = snapshot_root / topic
    topic_benchmark = benchmark_root / topic
    clean = pd.read_parquet(topic_input / "clean.parquet")
    clustered = pd.read_parquet(topic_input / "clustered.parquet")
    if not np.array_equal(clean["id"].to_numpy(), clustered["id"].to_numpy()):
        raise ValueError("Frozen clean and clustered records are not aligned")
    candidates = pd.read_parquet(topic_benchmark / "complete_sentence_candidates.parquet")
    texts = clean["text"].tolist()
    summary_sentences = bm.split_summary(bm.TOPICS[topic]["summary"])
    with open(topic_benchmark / "exact_random_windows.json") as handle:
        exact_audit = json.load(handle)["draws"]
    exact_texts = [item["text"] for draw in exact_audit for item in draw]
    combined_texts = texts + summary_sentences + candidates["text"].tolist() + exact_texts
    embedding_path = topic_benchmark / f"embeddings_{model_name}.npz"
    embeddings = bm.normalize(bm.get_embeddings(model_name, combined_texts, embedding_path))
    n_records = len(texts)
    n_summary = len(summary_sentences)
    n_candidates = len(candidates)
    record_embeddings = embeddings[:n_records]
    summary_embeddings = embeddings[n_records:n_records + n_summary]
    candidate_embeddings = embeddings[
        n_records + n_summary:n_records + n_summary + n_candidates
    ]
    with open(topic_benchmark / f"benchmark_{model_name}.json") as handle:
        exploratory_result = json.load(handle)
    rules = exploratory_result["candidate_length_rule"]
    if not (
        rules.get("cross_record_exact_duplicate_sentences_excluded") and
        rules.get("verbatim_consultation_prompt_sentences_excluded")
    ):
        raise ValueError("Benchmark root is not the leakage-controlled final candidate pool")
    return (
        clean,
        clustered,
        candidates,
        summary_sentences,
        record_embeddings,
        summary_embeddings,
        candidate_embeddings,
        exploratory_result,
    )


def run_crossfit_topic(
    topic: str,
    model_name: str,
    snapshot_root: Path,
    benchmark_root: Path,
    output_root: Path,
    n_repeats: int,
    n_folds: int,
    n_bootstrap: int,
):
    (
        clean,
        clustered,
        candidates,
        summary_sentences,
        record_embeddings,
        summary_embeddings,
        candidate_embeddings,
        exploratory_result,
    ) = load_verified_embeddings(topic, model_name, snapshot_root, benchmark_root)
    labels = clustered["cluster_id"].to_numpy(dtype=int)
    n_records = len(clean)
    budgets = [len(sentence.split()) for sentence in summary_sentences]
    record_summary_sim = record_embeddings @ summary_embeddings.T
    official_coverage = record_summary_sim.max(axis=1)
    threshold = float(official_coverage.mean() - official_coverage.std())
    similarity = record_embeddings @ candidate_embeddings.T
    owner_values = candidates["owner_index"].to_numpy(dtype=int)

    prediction_matrices = {
        method: np.full((n_repeats, n_records), np.nan, dtype=np.float32)
        for method in METHODS
    }
    prediction_matrices["official"][:] = official_coverage
    repeat_results = []
    fold_results = []
    repeat_rng = np.random.RandomState(SEED)
    repeat_seeds = repeat_rng.randint(0, 2**31 - 1, size=n_repeats)
    print(
        f"\n{topic}/{model_name}: n={n_records}, candidates={len(candidates)}, "
        f"{n_repeats}x{n_folds}-fold cross-fitting"
    )

    for repeat_index, repeat_seed in enumerate(repeat_seeds):
        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=int(repeat_seed),
        )
        test_counts = np.zeros(n_records, dtype=int)
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(n_records))
        ):
            test_counts[test_indices] += 1
            train_owner_mask = np.zeros(n_records, dtype=bool)
            train_owner_mask[train_indices] = True
            sim_train = similarity[train_indices]
            sim_test = similarity[test_indices]
            fold_seed = int(repeat_seed + 10007 * (fold_index + 1)) % (2**31 - 1)
            fold_rng = np.random.RandomState(fold_seed)

            slot_pools = bm.split_slot_candidates(
                candidates, train_owner_mask, budgets
            )
            random_expected = np.zeros(len(test_indices), dtype=np.float32)
            for _ in range(N_COMPLETE_RANDOM_PER_FOLD):
                selected_random = bm.sample_complete_summary(
                    slot_pools, owner_values, budgets, fold_rng
                )
                random_expected += sim_test[:, selected_random].max(axis=1)
            random_expected /= N_COMPLETE_RANDOM_PER_FOLD

            candidate_pool = bm.prefilter_candidates(
                sim_train, candidates, train_owner_mask
            )
            selected_mean = bm.greedy_select(
                sim_train, candidates, candidate_pool, budgets, "mean"
            )
            selected_tail = bm.greedy_select(
                sim_train, candidates, candidate_pool, budgets, "balanced_tail"
            )
            mean_coverage = sim_test[:, selected_mean].max(axis=1)
            tail_coverage = sim_test[:, selected_tail].max(axis=1)
            prediction_matrices["complete_sentence_random_expected"][
                repeat_index, test_indices
            ] = random_expected
            prediction_matrices["mean_optimized_extractive"][
                repeat_index, test_indices
            ] = mean_coverage
            prediction_matrices["balanced_tail_optimized_extractive"][
                repeat_index, test_indices
            ] = tail_coverage
            fold_results.append({
                "repeat": repeat_index,
                "fold": fold_index,
                "repeat_seed": int(repeat_seed),
                "fold_seed": fold_seed,
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "test_indices": test_indices.tolist(),
                "metrics": {
                    "official": scalar_metrics(official_coverage[test_indices], threshold),
                    "complete_sentence_random_expected": scalar_metrics(
                        random_expected, threshold
                    ),
                    "mean_optimized_extractive": scalar_metrics(mean_coverage, threshold),
                    "balanced_tail_optimized_extractive": scalar_metrics(
                        tail_coverage, threshold
                    ),
                },
                "mean_optimized_selection": bm.describe_selection(
                    selected_mean, candidates
                ),
                "balanced_tail_selection": bm.describe_selection(
                    selected_tail, candidates
                ),
            })
            print(
                f"  repeat {repeat_index + 1}/{n_repeats}, fold {fold_index + 1}/{n_folds}: "
                f"official {official_coverage[test_indices].mean():.3f}; "
                f"mean-opt {mean_coverage.mean():.3f}; "
                f"tail-opt tail {scalar_metrics(tail_coverage, threshold)['bottom_decile_mean']:.3f}"
            )
        if not np.all(test_counts == 1):
            raise AssertionError("Every participant must be tested exactly once per repetition")
        for method in METHODS[1:]:
            if np.isnan(prediction_matrices[method][repeat_index]).any():
                raise AssertionError(f"Missing OOF predictions for {method}")
        repeat_results.append({
            "repeat": repeat_index,
            "seed": int(repeat_seed),
            "participant_test_count_min": int(test_counts.min()),
            "participant_test_count_max": int(test_counts.max()),
            "metrics": {
                method: scalar_metrics(
                    prediction_matrices[method][repeat_index], threshold
                )
                for method in METHODS
            },
        })

    participant_predictions = {
        method: values.mean(axis=0)
        for method, values in prediction_matrices.items()
    }
    inference = paired_participant_bootstrap(
        participant_predictions,
        threshold,
        n_bootstrap,
        SEED + 700001,
    )
    repeat_sensitivity = repeat_level_sensitivity(
        prediction_matrices,
        threshold,
    )
    group_diagnostics = group_bootstrap_diagnostics(
        participant_predictions,
        labels,
        threshold,
        n_bootstrap,
        SEED + 900001,
    )

    topic_output = output_root / topic
    topic_output.mkdir(parents=True, exist_ok=True)
    prediction_frame = pd.DataFrame({
        "id": clean["id"].to_numpy(),
        "cluster_id": labels,
        **{
            f"{method}_mean_oof": values
            for method, values in participant_predictions.items()
        },
    })
    for method, values in prediction_matrices.items():
        for repeat_index in range(n_repeats):
            prediction_frame[f"{method}_repeat_{repeat_index + 1}"] = values[repeat_index]
    prediction_path = topic_output / f"participant_predictions_{model_name}.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)

    result = {
        "topic": topic,
        "model": model_name,
        "embedding_model": exploratory_result["embedding_model"],
        "frozen_input_n": n_records,
        "frozen_cluster_count": int(len(np.unique(labels))),
        "official_sentence_word_budgets": budgets,
        "official_total_words": int(sum(budgets)),
        "candidate_sentence_count": int(len(candidates)),
        "candidate_rules": exploratory_result["candidate_length_rule"],
        "frozen_exclusion_threshold": threshold,
        "crossfit_protocol": {
            "n_repeats": n_repeats,
            "n_folds": n_folds,
            "stratified_by_frozen_cluster": False,
            "fold_assignment": "shuffled KFold independent of full-corpus cluster labels",
            "shuffle_within_repetition": True,
            "each_participant_tested_once_per_repetition": True,
            "complete_random_summaries_per_fold": N_COMPLETE_RANDOM_PER_FOLD,
            "complete_random_role": (
                "Mean expected participant coverage diagnostic only; not included in "
                "participant-level tail inference because averaging random summaries "
                "changes the lower-tail estimand."
            ),
            "candidate_prefilter": "training-only mean and bottom-decile similarity",
            "selection_access": "training participant embeddings and their candidate sentences only",
            "evaluation_access": "held-out fold only",
            "participant_estimand": "mean out-of-fold coverage across repetitions",
            "limitation": (
                "Full-corpus clusters are used only for post hoc analytic-region diagnostics. "
                "They do not determine fold assignment, candidate prefiltering, or selection. "
                "Bootstrap intervals condition on the fitted crossfit selections and generated "
                "embeddings; they do not refit selection or regenerate embeddings in each resample."
            ),
        },
        "exact_length_random_reference": exploratory_result["exact_length_random"],
        "participant_level_bootstrap": inference,
        "repeat_level_sensitivity": repeat_sensitivity,
        "group_bootstrap_diagnostics": group_diagnostics,
        "repeat_results": repeat_results,
        "fold_results": fold_results,
        "participant_predictions_file": prediction_path.name,
    }
    output_path = topic_output / f"crossfit_{model_name}.json"
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, cls=bm.NumpyEncoder)
    print(f"  saved {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=bm.OUTPUT_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=bm.OUTPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=bm.OUTPUT_DIR)
    parser.add_argument("--model", choices=["mpnet", "openai"], required=True)
    parser.add_argument("--topic", choices=list(bm.TOPICS) + ["all"], default="all")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    args = parser.parse_args()
    if args.repeats < 2 or args.folds < 2 or args.bootstrap < 1000:
        raise ValueError("Use at least 2 repeats, 2 folds, and 1000 bootstrap draws")
    args.output_root.mkdir(parents=True, exist_ok=True)
    topics = list(bm.TOPICS) if args.topic == "all" else [args.topic]
    for topic in topics:
        run_crossfit_topic(
            topic,
            args.model,
            args.snapshot_root,
            args.benchmark_root,
            args.output_root,
            args.repeats,
            args.folds,
            args.bootstrap,
        )


if __name__ == "__main__":
    main()
