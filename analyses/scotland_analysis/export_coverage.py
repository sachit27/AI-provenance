#!/usr/bin/env python3
"""Export per-participant coverage in a flat, model-stacked contract.

For every topic directory ``<analysis-root>/<topic>/`` writes
``<out-root>/<topic>/participant_coverage.csv`` with columns

    respondent_id, embedding_model, coverage, low_coverage, region, word_count

* respondent_id    : clean.parquet ``Internal ID`` (the source respondent ID; the
                     pipeline's ``id`` column is "P" + this value)
* embedding_model  : openai | mpnet | nomic
* coverage         : max cosine similarity between the participant's embedding
                     and the official-summary sentence embeddings (exactly
                     04_transport.compute_coverage / 06_cross_topic definition)
* low_coverage     : coverage < mean(coverage) - std(coverage) (population SD),
                     the pipeline's mu-sigma threshold, computed per model
* region           : cluster_id from clustered.parquet (K-means on the OpenAI
                     embeddings, 03_topology.py); the same partition is used
                     for all three models, as in 06_cross_topic.py
* word_count       : clean.parquet word_count

Read-only on ``--analysis-root``.  Works for Scotland (default) and Canada:

    python analyses/scotland_analysis/export_coverage.py
    python analyses/scotland_analysis/export_coverage.py \
        --analysis-root src/analysis --topics education trust \
        --out-root analyses/canada_coverage
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = ("openai", "mpnet", "nomic")
COLUMNS = ["respondent_id", "embedding_model", "coverage", "low_coverage", "region", "word_count"]


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-10)


def coverage_scores(records: np.ndarray, summary: np.ndarray) -> np.ndarray:
    """Identical to 04_transport.compute_coverage."""
    return (_normalise(records) @ _normalise(summary).T).max(axis=1)


def n_summary_sentences(base: Path) -> int | None:
    transport = base / "transport.json"
    if transport.exists():
        return int(json.loads(transport.read_text())["n_summary_sentences"])
    return None


def load_blocks(base: Path, model: str, n_records: int):
    if model == "nomic":
        path = base / "embeddings_nomic_embed_text.npz"
        if not path.exists():
            return None
        saved = np.load(path, allow_pickle=True)
        return saved["input_embeddings"], saved["summary_embeddings"]
    path = base / f"embeddings_{model}.npz"
    if not path.exists():
        return None
    embeddings = np.load(path)["embeddings"]
    n_summary = n_summary_sentences(base)
    if n_summary is None:
        candidates = pd.read_parquet(base / "complete_sentence_candidates.parquet")
        draws = json.loads((base / "exact_random_windows.json").read_text())["draws"]
        n_summary = len(embeddings) - n_records - len(candidates) - sum(len(d) for d in draws)
    return embeddings[:n_records], embeddings[n_records:n_records + n_summary]


def export_topic(base: Path, out_dir: Path, topic: str, models) -> None:
    clean = pd.read_parquet(base / "clean.parquet")
    n_records = len(clean)
    if "Internal ID" in clean.columns:
        respondent = clean["Internal ID"].astype(str).to_numpy()
    else:
        respondent = clean["id"].astype(str).str.replace(r"^P", "", regex=True).to_numpy()
    region = np.full(n_records, np.nan)
    primary_cov = None
    clustered_path = base / "clustered.parquet"
    if clustered_path.exists():
        clustered = pd.read_parquet(clustered_path)
        lookup = clustered.set_index("id")
        if "cluster_id" in lookup.columns and set(clean["id"]) <= set(lookup.index):
            region = lookup.loc[clean["id"], "cluster_id"].to_numpy()
            if "coverage_score" in lookup.columns:
                primary_cov = lookup.loc[clean["id"], "coverage_score"].to_numpy(dtype=float)
    frames, thresholds = [], {}
    for model in models:
        blocks = load_blocks(base, model, n_records)
        if blocks is None:
            print(f"  {topic}/{model}: embeddings not found; skipped")
            continue
        records, summary = blocks
        if len(records) != n_records:
            raise ValueError(f"{topic}/{model}: {len(records)} embeddings for {n_records} records")
        coverage = coverage_scores(records, summary)
        if model == "openai" and primary_cov is not None:
            if not np.allclose(coverage, primary_cov, atol=1e-4):
                print(f"  WARNING {topic}/openai: recomputed coverage differs from clustered.parquet")
        threshold = float(coverage.mean() - coverage.std())
        thresholds[model] = {"threshold_mean_minus_sd": threshold,
                             "n_summary_sentences": int(len(summary)),
                             "n_participants": int(n_records)}
        frames.append(pd.DataFrame({
            "respondent_id": respondent,
            "embedding_model": model,
            "coverage": coverage,
            "low_coverage": coverage < threshold,
            "region": region,
            "word_count": clean["word_count"].to_numpy(),
        })[COLUMNS])
    if not frames:
        print(f"  {topic}: nothing exported")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True)
    if frame["region"].notna().all():
        frame["region"] = frame["region"].astype(int)
    frame.to_csv(out_dir / "participant_coverage.csv", index=False)
    (out_dir / "participant_coverage_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    print(f"  {topic}: {len(frame)} rows -> {out_dir / 'participant_coverage.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, default=HERE / "analysis")
    parser.add_argument("--topics", nargs="+")
    parser.add_argument("--out-root", type=Path, help="default: same as --analysis-root")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = parser.parse_args()
    root = args.analysis_root.expanduser().resolve()
    out_root = (args.out_root or args.analysis_root).expanduser().resolve()
    topics = args.topics or sorted(p.name for p in root.iterdir()
                                   if (p / "clean.parquet").exists())
    if not topics:
        parser.error(f"no <topic>/clean.parquet under {root}")
    for topic in topics:
        export_topic(root / topic, out_root / topic, topic, args.models)


if __name__ == "__main__":
    main()
