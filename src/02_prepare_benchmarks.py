#!/usr/bin/env python3
"""Held-out, budget-matched benchmarks for consultation-summary coverage.

This script asks three distinct questions on the versioned analysis corpus:

1. Exact-length random windows: is the official summary better than chance
   when the six per-sentence word budgets are matched exactly?
2. Complete-sentence random summaries: does that conclusion survive when the
   random units are readable participant sentences rather than word windows?
3. Held-out optimized extractive summaries: how does the official summary
   compare with six participant sentences selected on an 80% training split
   and evaluated on the untouched 20% test split?

Two extractive objectives are reported: mean semantic coverage and a balanced
objective averaging mean coverage with bottom-decile coverage. Selection and
evaluation use the same embedding model within a run. Results are repeated in
OpenAI text-embedding-3-large and local all-mpnet-base-v2 spaces.

The optimized outputs are coverage benchmarks, not publication-ready prose.
They may lack narrative coherence because no coherence objective is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    EMBEDDING_BATCH_SIZE,
    NumpyEncoder,
    OPENAI_API_KEY,
    OUTPUT_DIR,
    OPENAI_EMBEDDING_MODEL,
    TOPICS,
)

SEED = 20260717
N_SPLITS = 20
TEST_FRACTION = 0.20
N_COMPLETE_RANDOM_PER_SPLIT = 100
N_EXACT_RANDOM = 1000
MIN_SENTENCE_WORDS = 6
SLOT_LENGTH_LOWER = 0.75
# Complete participant sentences may be shorter than a slot, but may never
# exceed the corresponding official-summary word budget. This makes the
# extractive comparison conservative and prevents it from buying coverage
# with additional words.
SLOT_LENGTH_UPPER = 1.00
PREFILTER_TOP_MEAN = 800
PREFILTER_TOP_TAIL = 800
TAIL_FRACTION = 0.10


def split_summary(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text.strip())
    return [part.strip() for part in parts if part.strip()]


def split_participant_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+|[\r\n]+', str(text).strip())
    return [part.strip() for part in parts if part.strip()]


def canonical_sentence(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-10)


def gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if np.any(values < 0):
        raise ValueError("Coverage Gini requires non-negative values")
    if len(values) < 2 or values.sum() == 0:
        return 0.0
    index = np.arange(1, len(values) + 1)
    return float(np.sum((2 * index - len(values) - 1) * values) /
                 (len(values) * values.sum()))


def build_candidates(df: pd.DataFrame, max_words: int,
                     excluded_sentences: set[str] | None = None) -> pd.DataFrame:
    excluded_sentences = excluded_sentences or set()
    occurrences = []
    for owner, text in enumerate(df["text"].tolist()):
        for sentence in split_participant_sentences(text):
            words = sentence.split()
            if not MIN_SENTENCE_WORDS <= len(words) <= max_words:
                continue
            key = canonical_sentence(sentence)
            if key in excluded_sentences:
                continue
            occurrences.append((key, owner, len(words), sentence))
    counts = Counter(item[0] for item in occurrences)
    records = []
    # Repeated wording is excluded entirely so an exact duplicate cannot occur
    # on both sides of a participant-level train/test split.
    for key, owner, word_count, sentence in occurrences:
        if counts[key] != 1:
            continue
        records.append({
            "candidate_id": len(records),
            "owner_index": owner,
            "word_count": word_count,
            "text": sentence,
        })
    return pd.DataFrame(records)


def embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for --model openai")
    client = OpenAI(api_key=OPENAI_API_KEY)
    output = []
    for lo in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = [text[:8000] for text in texts[lo:lo + EMBEDDING_BATCH_SIZE]]
        response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        output.extend(item.embedding for item in response.data)
        if (lo // EMBEDDING_BATCH_SIZE + 1) % 10 == 0:
            print(f"    embedded {min(lo + EMBEDDING_BATCH_SIZE, len(texts)):,}/{len(texts):,}")
    return np.asarray(output, dtype=np.float32)


def embed_mpnet(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    return np.asarray(model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    ), dtype=np.float32)


def text_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for value in texts:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def get_embeddings(model_name: str, texts: list[str], embedding_path: Path) -> np.ndarray:
    """Load the versioned embedding artifact or create it once from its texts."""
    expected_digest = text_digest(texts)
    if embedding_path.exists():
        saved = np.load(embedding_path)
        embeddings = saved["embeddings"]
        if len(embeddings) != len(texts):
            raise ValueError(f"Embedding artifact length mismatch: {embedding_path}")
        if "text_digest" not in saved or str(saved["text_digest"]) != expected_digest:
            raise ValueError(f"Embedding artifact text digest mismatch: {embedding_path}")
        return embeddings
    embeddings = embed_openai(texts) if model_name == "openai" else embed_mpnet(texts)
    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        text_digest=np.asarray(expected_digest),
    )
    return embeddings


def exact_random_windows(texts: list[str], budgets: list[int], n_draws: int, seed: int):
    tokenized = [str(text).split() for text in texts]
    eligible = {
        budget: np.array([i for i, words in enumerate(tokenized) if len(words) >= budget])
        for budget in set(budgets)
    }
    rng = np.random.RandomState(seed)
    windows, audit = [], []
    for draw in range(n_draws):
        used = set()
        draw_audit = []
        for slot, budget in enumerate(budgets):
            available = np.array([i for i in eligible[budget] if int(i) not in used])
            owner = int(rng.choice(available if len(available) else eligible[budget]))
            used.add(owner)
            words = tokenized[owner]
            start = int(rng.randint(0, len(words) - budget + 1))
            text = " ".join(words[start:start + budget])
            windows.append(text)
            draw_audit.append({
                "slot": slot, "owner_index": owner, "start_word": start,
                "word_count": budget, "text": text,
            })
        audit.append(draw_audit)
    return windows, audit


def coverage_metrics(coverage: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    coverage = np.asarray(coverage, dtype=float)
    tail_n = max(1, int(math.ceil(TAIL_FRACTION * len(coverage))))
    lower_tail = np.partition(coverage, tail_n - 1)[:tail_n]
    per_group = {}
    for label in sorted(np.unique(labels)):
        group = coverage[labels == label]
        per_group[str(int(label))] = {
            "n": int(len(group)),
            "mean_coverage": float(group.mean()),
            "exclusion_rate_at_frozen_threshold": float(np.mean(group < threshold)),
        }
    return {
        "mean_coverage": float(coverage.mean()),
        "median_coverage": float(np.median(coverage)),
        "p10_coverage": float(np.quantile(coverage, 0.10)),
        "bottom_decile_mean": float(lower_tail.mean()),
        "gini": gini(coverage),
        "exclusion_rate_at_frozen_threshold": float(np.mean(coverage < threshold)),
        "minimum_group_mean_coverage": min(v["mean_coverage"] for v in per_group.values()),
        "maximum_group_exclusion_rate": max(
            v["exclusion_rate_at_frozen_threshold"] for v in per_group.values()
        ),
        "per_group": per_group,
    }


def slot_candidates(candidates: pd.DataFrame, budget: int, allowed_mask: np.ndarray) -> np.ndarray:
    lower = max(MIN_SENTENCE_WORDS, int(math.floor(SLOT_LENGTH_LOWER * budget)))
    upper = int(math.ceil(SLOT_LENGTH_UPPER * budget))
    lengths = candidates["word_count"].to_numpy()
    return np.flatnonzero(allowed_mask & (lengths >= lower) & (lengths <= upper))


def split_slot_candidates(candidates, train_owner_mask, budgets):
    owner_values = candidates["owner_index"].to_numpy(dtype=int)
    allowed = train_owner_mask[owner_values]
    return {
        budget: slot_candidates(candidates, budget, allowed)
        for budget in set(budgets)
    }


def sample_complete_summary(slot_pools, owner_values, budgets, rng):
    selected, used_owners = [], set()
    for budget in sorted(budgets, reverse=True):
        eligible = slot_pools[budget]
        if used_owners:
            eligible = eligible[~np.isin(owner_values[eligible], list(used_owners))]
        if not len(eligible):
            raise RuntimeError("No complete sentence satisfies a length slot")
        choice = int(rng.choice(eligible))
        selected.append(choice)
        used_owners.add(int(owner_values[choice]))
    return selected


def prefilter_candidates(sim_train, candidates, train_owner_mask):
    """Shortlist candidates using training records only.

    The prefilter deliberately avoids cluster labels because the manuscript's
    analytic partition is fitted to the complete corpus. Using those labels in
    held-out selection would allow test embeddings to influence the candidate
    pool indirectly.
    """
    allowed = train_owner_mask[candidates["owner_index"].to_numpy()]
    allowed_idx = np.flatnonzero(allowed)
    local = sim_train[:, allowed_idx]
    mean_score = local.mean(axis=0)
    tail_n = max(1, int(math.ceil(TAIL_FRACTION * local.shape[0])))
    tail_score = np.partition(local, tail_n - 1, axis=0)[:tail_n].mean(axis=0)
    keep_local = set(np.argsort(mean_score)[-PREFILTER_TOP_MEAN:].tolist())
    keep_local.update(np.argsort(tail_score)[-PREFILTER_TOP_TAIL:].tolist())
    return allowed_idx[np.array(sorted(keep_local), dtype=int)]


def greedy_select(sim_train_all, candidates, pool, budgets, objective):
    current = np.zeros(sim_train_all.shape[0], dtype=np.float32)
    selected, used_owners = [], set()
    owner_values = candidates["owner_index"].to_numpy()
    lengths = candidates["word_count"].to_numpy()
    for budget in sorted(budgets, reverse=True):
        lower = max(MIN_SENTENCE_WORDS, int(math.floor(SLOT_LENGTH_LOWER * budget)))
        upper = int(math.ceil(SLOT_LENGTH_UPPER * budget))
        feasible = np.array([
            idx for idx in pool
            if lower <= lengths[idx] <= upper and int(owner_values[idx]) not in used_owners
        ], dtype=int)
        if not len(feasible):
            raise RuntimeError(f"No feasible optimized candidate for {budget}-word slot")
        new_coverage = np.maximum(current[:, None], sim_train_all[:, feasible])
        means = new_coverage.mean(axis=0)
        if objective == "mean":
            scores = means
        elif objective == "balanced_tail":
            tail_n = max(1, int(math.ceil(TAIL_FRACTION * len(current))))
            tails = np.partition(new_coverage, tail_n - 1, axis=0)[:tail_n].mean(axis=0)
            scores = 0.5 * means + 0.5 * tails
        else:
            raise ValueError(objective)
        winner = int(feasible[int(np.argmax(scores))])
        selected.append(winner)
        used_owners.add(int(owner_values[winner]))
        current = np.maximum(current, sim_train_all[:, winner])
    return selected


def describe_selection(selected, candidates):
    rows = candidates.iloc[selected]
    return {
        "total_words": int(rows["word_count"].sum()),
        "sentences": [
            {
                "candidate_id": int(row.candidate_id),
                "owner_index": int(row.owner_index),
                "word_count": int(row.word_count),
                "text": row.text,
            }
            for row in rows.itertuples(index=False)
        ],
    }


def aggregate_split_results(split_results: list[dict], method: str) -> dict:
    metric_keys = [
        "mean_coverage", "median_coverage", "p10_coverage",
        "bottom_decile_mean", "gini", "exclusion_rate_at_frozen_threshold",
        "minimum_group_mean_coverage", "maximum_group_exclusion_rate",
    ]
    aggregate = {}
    for key in metric_keys:
        values = np.array([split[method][key] for split in split_results], dtype=float)
        aggregate[key] = {
            "mean_across_splits": float(values.mean()),
            "sd_across_splits": float(values.std(ddof=1)),
            "range_across_splits": [float(values.min()), float(values.max())],
        }
    if method != "official":
        differences = {}
        for key in metric_keys:
            candidate = np.array([split[method][key] for split in split_results])
            official = np.array([split["official"][key] for split in split_results])
            diff = candidate - official
            higher_is_better = key not in {
                "gini", "exclusion_rate_at_frozen_threshold", "maximum_group_exclusion_rate"
            }
            better = diff > 0 if higher_is_better else diff < 0
            differences[key] = {
                "mean_paired_difference": float(diff.mean()),
                "percentile_interval_2.5_97.5": [
                    float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
                ],
                "fraction_splits_better_than_official": float(better.mean()),
            }
        aggregate["paired_vs_official"] = differences
    return aggregate


def prepare_topic_inputs(topic, snapshot_root: Path, output_root: Path,
                         model_name: str):
    """Build and validate the one combined embedding corpus for a topic/model.

    This stage deliberately does not require cluster assignments, so a clean
    regeneration can construct the authoritative participant/summary rows
    before topology, coverage, or cross-fitting is run.
    """
    topic_input = snapshot_root / topic
    topic_output = output_root / topic
    topic_output.mkdir(parents=True, exist_ok=True)
    clean = pd.read_parquet(topic_input / "clean.parquet")
    texts = clean["text"].tolist()
    summary_sentences = split_summary(TOPICS[topic]["summary"])
    budgets = [len(sentence.split()) for sentence in summary_sentences]
    if len(budgets) != 6:
        raise ValueError("Expected six official summary sentences")

    candidate_path = topic_output / "complete_sentence_candidates.parquet"
    if candidate_path.exists():
        candidates = pd.read_parquet(candidate_path)
    else:
        prompt_columns = list(pd.read_csv(TOPICS[topic]["csv"], nrows=0).columns[1:])
        prompt_sentences = {
            canonical_sentence(sentence)
            for prompt in prompt_columns
            for sentence in split_participant_sentences(prompt)
        }
        candidates = build_candidates(
            clean,
            max_words=int(math.ceil(max(budgets) * SLOT_LENGTH_UPPER)),
            excluded_sentences=prompt_sentences,
        )
        candidates.to_parquet(candidate_path, index=False)
    print(f"\n{topic}/{model_name}: n={len(clean)}, candidates={len(candidates)}, budgets={budgets}")

    exact_path = topic_output / "exact_random_windows.json"
    if exact_path.exists():
        with open(exact_path) as f:
            exact_audit = json.load(f)["draws"]
        exact_texts = [item["text"] for draw in exact_audit for item in draw]
    else:
        exact_texts, exact_audit = exact_random_windows(texts, budgets, N_EXACT_RANDOM, SEED)
        with open(exact_path, "w") as f:
            json.dump({"budgets": budgets, "draws": exact_audit}, f)

    combined_texts = texts + summary_sentences + candidates["text"].tolist() + exact_texts
    embedding_path = topic_output / f"embeddings_{model_name}.npz"
    embeddings = normalize(get_embeddings(model_name, combined_texts, embedding_path))
    n_records = len(texts)
    n_summary = len(summary_sentences)
    n_candidates = len(candidates)
    record_embeddings = embeddings[:n_records]
    summary_embeddings = embeddings[n_records:n_records + n_summary]
    candidate_embeddings = embeddings[
        n_records + n_summary:n_records + n_summary + n_candidates
    ]
    exact_embeddings = embeddings[n_records + n_summary + n_candidates:]
    return (
        clean, candidates, summary_sentences, budgets, record_embeddings,
        summary_embeddings, candidate_embeddings, exact_embeddings, exact_audit,
    )


def evaluate_topic(topic, snapshot_root: Path, output_root: Path, model_name: str):
    (
        clean, candidates, summary_sentences, budgets, record_embeddings,
        summary_embeddings, candidate_embeddings, exact_embeddings, exact_audit,
    ) = prepare_topic_inputs(
        topic, snapshot_root, output_root, model_name
    )
    topic_input = snapshot_root / topic
    topic_output = output_root / topic
    clustered = pd.read_parquet(topic_input / "clustered.parquet")
    if not np.array_equal(clean["id"].to_numpy(), clustered["id"].to_numpy()):
        raise ValueError("Clean and clustered records are not aligned")
    labels = clustered["cluster_id"].to_numpy(dtype=int)
    n_records = len(clean)
    n_candidates = len(candidates)
    record_summary_sim = record_embeddings @ summary_embeddings.T
    official_full_coverage = record_summary_sim.max(axis=1)
    frozen_threshold = float(official_full_coverage.mean() - official_full_coverage.std())
    official_full = coverage_metrics(official_full_coverage, labels, frozen_threshold)

    # Full-corpus exact-length chance null.
    exact_scores = []
    for draw in range(N_EXACT_RANDOM):
        selected = exact_embeddings[draw * 6:(draw + 1) * 6]
        coverage = (record_embeddings @ selected.T).max(axis=1)
        exact_scores.append(coverage_metrics(coverage, labels, frozen_threshold))
    exact_mean = np.array([result["mean_coverage"] for result in exact_scores])
    exact_tail = np.array([result["bottom_decile_mean"] for result in exact_scores])
    exact_null = {
        "n_draws": N_EXACT_RANDOM,
        "official": official_full,
        "random_mean_coverage": {
            "mean": float(exact_mean.mean()), "sd": float(exact_mean.std()),
            "official_upper_tail_p": float((1 + np.sum(exact_mean >= official_full["mean_coverage"])) /
                                           (N_EXACT_RANDOM + 1)),
            "official_lower_tail_p": float((1 + np.sum(exact_mean <= official_full["mean_coverage"])) /
                                           (N_EXACT_RANDOM + 1)),
        },
        "random_bottom_decile_mean": {
            "mean": float(exact_tail.mean()), "sd": float(exact_tail.std()),
            "official_upper_tail_p": float((1 + np.sum(exact_tail >= official_full["bottom_decile_mean"])) /
                                           (N_EXACT_RANDOM + 1)),
            "official_lower_tail_p": float((1 + np.sum(exact_tail <= official_full["bottom_decile_mean"])) /
                                           (N_EXACT_RANDOM + 1)),
        },
    }

    # Precompute record-to-candidate similarity once.
    similarity = record_embeddings @ candidate_embeddings.T
    owner_values = candidates["owner_index"].to_numpy(dtype=int)
    splitter_rng = np.random.RandomState(SEED)
    split_seeds = splitter_rng.randint(0, 2**31 - 1, size=N_SPLITS)
    split_results = []
    for split_index, split_seed in enumerate(split_seeds):
        rng = np.random.RandomState(int(split_seed))
        shuffled = rng.permutation(n_records)
        n_test = max(1, int(round(TEST_FRACTION * n_records)))
        test_indices = np.sort(shuffled[:n_test])
        train_indices = np.sort(shuffled[n_test:])
        train_owner_mask = np.zeros(n_records, dtype=bool)
        train_owner_mask[train_indices] = True
        sim_train = similarity[train_indices]
        sim_test = similarity[test_indices]

        official_test_coverage = record_summary_sim[test_indices].max(axis=1)
        official_test = coverage_metrics(
            official_test_coverage, labels[test_indices], frozen_threshold
        )

        random_metrics = []
        slot_pools = split_slot_candidates(candidates, train_owner_mask, budgets)
        for _ in range(N_COMPLETE_RANDOM_PER_SPLIT):
            selected = sample_complete_summary(
                slot_pools, owner_values, budgets, rng
            )
            cov = sim_test[:, selected].max(axis=1)
            random_metrics.append(coverage_metrics(cov, labels[test_indices], frozen_threshold))
        random_average = {
            key: float(np.mean([result[key] for result in random_metrics]))
            for key in random_metrics[0]
            if key != "per_group"
        }
        random_mean_values = np.array([
            result["mean_coverage"] for result in random_metrics
        ])
        random_tail_values = np.array([
            result["bottom_decile_mean"] for result in random_metrics
        ])
        complete_random_null_test = {
            "n_draws": N_COMPLETE_RANDOM_PER_SPLIT,
            "mean_coverage": {
                "official_upper_tail_p": float(
                    (1 + np.sum(random_mean_values >= official_test["mean_coverage"])) /
                    (N_COMPLETE_RANDOM_PER_SPLIT + 1)
                ),
                "random_mean": float(random_mean_values.mean()),
            },
            "bottom_decile_mean": {
                "official_upper_tail_p": float(
                    (1 + np.sum(random_tail_values >= official_test["bottom_decile_mean"])) /
                    (N_COMPLETE_RANDOM_PER_SPLIT + 1)
                ),
                "random_mean": float(random_tail_values.mean()),
            },
        }

        candidate_pool = prefilter_candidates(sim_train, candidates, train_owner_mask)
        selected_mean = greedy_select(
            sim_train, candidates, candidate_pool, budgets, "mean"
        )
        selected_tail = greedy_select(
            sim_train, candidates, candidate_pool, budgets, "balanced_tail"
        )
        mean_test = coverage_metrics(
            sim_test[:, selected_mean].max(axis=1), labels[test_indices], frozen_threshold
        )
        tail_test = coverage_metrics(
            sim_test[:, selected_tail].max(axis=1), labels[test_indices], frozen_threshold
        )
        split_results.append({
            "split": split_index,
            "seed": int(split_seed),
            "n_train": int(len(train_indices)),
            "n_test": int(len(test_indices)),
            "official": official_test,
            "complete_sentence_random": random_average,
            "complete_sentence_random_null_test": complete_random_null_test,
            "mean_optimized_extractive": mean_test,
            "balanced_tail_optimized_extractive": tail_test,
            "mean_optimized_selection": describe_selection(selected_mean, candidates),
            "balanced_tail_selection": describe_selection(selected_tail, candidates),
        })
        print(
            f"  split {split_index + 1:02d}/{N_SPLITS}: official mean/tail "
            f"{official_test['mean_coverage']:.3f}/{official_test['bottom_decile_mean']:.3f}; "
            f"mean-opt {mean_test['mean_coverage']:.3f}/{mean_test['bottom_decile_mean']:.3f}; "
            f"tail-opt {tail_test['mean_coverage']:.3f}/{tail_test['bottom_decile_mean']:.3f}"
        )

    methods = [
        "official", "complete_sentence_random", "mean_optimized_extractive",
        "balanced_tail_optimized_extractive",
    ]
    complete_random_p_values = {
        metric: np.array([
            split["complete_sentence_random_null_test"][metric]["official_upper_tail_p"]
            for split in split_results
        ])
        for metric in ["mean_coverage", "bottom_decile_mean"]
    }
    result = {
        "topic": topic,
        "model": model_name,
        "embedding_model": OPENAI_EMBEDDING_MODEL if model_name == "openai" else "all-mpnet-base-v2",
        "frozen_input_n": n_records,
        "frozen_cluster_count": int(len(np.unique(labels))),
        "official_sentence_word_budgets": budgets,
        "official_total_words": int(sum(budgets)),
        "candidate_sentence_count": int(len(candidates)),
        "candidate_length_rule": {
            "minimum_words": MIN_SENTENCE_WORDS,
            "slot_lower_multiplier": SLOT_LENGTH_LOWER,
            "slot_upper_multiplier": SLOT_LENGTH_UPPER,
            "cross_record_exact_duplicate_sentences_excluded": True,
            "verbatim_consultation_prompt_sentences_excluded": True,
        },
        "frozen_exclusion_threshold": frozen_threshold,
        "exact_length_random": exact_null,
        "held_out_protocol": {
            "n_splits": N_SPLITS,
            "test_fraction": TEST_FRACTION,
            "stratified_by_frozen_cluster": False,
            "complete_random_draws_per_split": N_COMPLETE_RANDOM_PER_SPLIT,
            "selection_access": "training records and their candidate sentences only",
            "evaluation_access": "untouched held-out records",
            "optimized_objectives": {
                "mean": "mean participant semantic coverage",
                "balanced_tail": "0.5 * mean coverage + 0.5 * bottom-decile mean coverage",
            },
            "fold_assignment": "random 80/20 split independent of full-corpus cluster labels",
            "candidate_prefilter": "training-only mean and bottom-decile similarity",
            "limitation": (
                "Full-corpus clusters are used only to describe held-out results by analytic region; "
                "they do not determine split membership or candidate selection."
            ),
        },
        "aggregate": {
            method: aggregate_split_results(split_results, method) for method in methods
        },
        "complete_sentence_random_null_across_splits": {
            metric: {
                "median_official_upper_tail_p": float(np.median(values)),
                "range": [float(values.min()), float(values.max())],
                "fraction_splits_p_at_most_0.05": float(np.mean(values <= 0.05)),
                "note": "Repeated splits overlap; these are stability diagnostics, not independent tests.",
            }
            for metric, values in complete_random_p_values.items()
        },
        "split_results": split_results,
    }
    output_path = topic_output / f"benchmark_{model_name}.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"  saved {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-root", type=Path,
                        default=OUTPUT_DIR)
    parser.add_argument("--model", choices=["mpnet", "openai"], required=True)
    parser.add_argument("--topic", choices=list(TOPICS) + ["all"], default="all")
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Build the candidate/null corpus and combined embeddings without requiring clusters",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    topics = list(TOPICS) if args.topic == "all" else [args.topic]
    for topic in topics:
        if args.prepare_only:
            prepare_topic_inputs(
                topic, args.snapshot_root, args.output_root, args.model,
            )
        else:
            evaluate_topic(
                topic, args.snapshot_root, args.output_root, args.model,
            )


if __name__ == "__main__":
    main()
