"""Validated access to the single analysis-ready embedding set.

Each topic has one combined embedding file per benchmarked model.  Its rows
are ordered as participant records, the six official-summary sentences,
complete-sentence candidates, and exact-length null windows.  Core coverage,
topology, chance, and cross-fitted analyses all read the same participant and
official-summary rows from these files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, TOPICS


def split_summary(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text.strip())
    return [part.strip() for part in parts if part.strip()]


def embedding_blocks(topic: str, model: str = "openai", root: Path | None = None):
    """Return validated record, summary, candidate, and null embedding blocks."""
    base = (root or OUTPUT_DIR) / topic
    clean = pd.read_parquet(base / "clean.parquet")
    candidates = pd.read_parquet(base / "complete_sentence_candidates.parquet")
    with open(base / "exact_random_windows.json") as handle:
        draws = json.load(handle)["draws"]
    summary = split_summary(TOPICS[topic]["summary"])
    if len(summary) != 6:
        raise ValueError(f"{topic}: expected six official-summary sentences")

    saved = np.load(base / f"embeddings_{model}.npz")
    embeddings = np.asarray(saved["embeddings"], dtype=np.float32)
    n_records = len(clean)
    n_summary = len(summary)
    n_candidates = len(candidates)
    n_null = sum(len(draw) for draw in draws)
    expected = n_records + n_summary + n_candidates + n_null
    if len(embeddings) != expected:
        raise ValueError(
            f"{topic}/{model}: {len(embeddings)} embeddings, expected {expected}"
        )
    return {
        "clean": clean,
        "candidates": candidates,
        "summary_sentences": summary,
        "null_draws": draws,
        "records": embeddings[:n_records],
        "summary": embeddings[n_records:n_records + n_summary],
        "candidates_embeddings": embeddings[
            n_records + n_summary:n_records + n_summary + n_candidates
        ],
        "null_embeddings": embeddings[n_records + n_summary + n_candidates:],
    }


def core_embeddings(topic: str, model: str = "openai", root: Path | None = None):
    blocks = embedding_blocks(topic, model, root)
    return blocks["records"], blocks["summary"], blocks["summary_sentences"]


def nomic_core_embeddings(topic: str, root: Path | None = None):
    base = (root or OUTPUT_DIR) / topic
    saved = np.load(base / "embeddings_nomic_embed_text.npz", allow_pickle=True)
    return saved["input_embeddings"], saved["summary_embeddings"]
