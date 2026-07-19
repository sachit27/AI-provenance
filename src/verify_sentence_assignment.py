#!/usr/bin/env python3
"""Generate and verify sentence_assignment.json.

Default mode verifies sentence_assignment.json in the configured analysis
directory against the analysis-ready artifacts. ``--generate`` (re)creates
the artifacts from those same inputs first, then verifies them, so the
artifact is fully reproducible from the released pipeline.

Contents per topic:
  - nearest-official-sentence assignment shares and assigned-similarity
    means (text-embedding-3-large space);
  - worst-covered-cluster diagnostics, including that cluster's mean
    similarity to EACH official sentence (all records, and among records
    assigned to that sentence);
  - source-cluster shares of the cross-fitted benchmark selections,
    zero-filled over all clusters so absences (e.g., zero selections from
    Education C8) are explicit and asserted.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS = ROOT / "outputs" / "analysis"
ANALYSIS = Path(os.environ.get(
    "PROVENANCE_OUTPUT_DIR", DEFAULT_ANALYSIS
)).expanduser().resolve()
WORST = {"education": 8, "trust": 4}


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def compute(topic: str) -> dict:
    worst = WORST[topic]
    df = pd.read_parquet(ANALYSIS / topic / "clustered.parquet")
    n = len(df)
    emb = np.load(ANALYSIS / topic / "embeddings_openai.npz",
                  allow_pickle=True)["embeddings"]
    ie, se = emb[:n], emb[n:n + 6]
    ien = ie / np.linalg.norm(ie, axis=1, keepdims=True)
    sen = se / np.linalg.norm(se, axis=1, keepdims=True)
    sims = ien @ sen.T
    nearest = sims.argmax(axis=1)
    cov = sims.max(axis=1)

    per_sentence = {}
    for j in range(6):
        m = nearest == j
        per_sentence[f"S{j+1}"] = {
            "nearest_share": round(float(m.mean()), 4),
            "mean_similarity_of_assigned":
                round(float(cov[m].mean()), 4) if m.any() else None,
            "n_assigned": int(m.sum())}

    wm = df.cluster_id.values == worst
    worst_per_sentence = {}
    for j in range(6):
        aj = wm & (nearest == j)
        worst_per_sentence[f"S{j+1}"] = {
            "n_assigned": int(aj.sum()),
            "mean_similarity_all_records": round(float(sims[wm, j].mean()), 4),
            "mean_similarity_among_assigned":
                round(float(sims[aj, j].mean()), 4) if aj.any() else None}

    clusters = sorted(int(c) for c in df.cluster_id.unique())
    sel = {}
    for model in ["openai", "mpnet"]:
        with (ANALYSIS / topic / f"crossfit_{model}.json").open() as handle:
            cj = json.load(handle)
        for key, name in [("mean_optimized_selection", "mean_objective"),
                          ("balanced_tail_selection", "tail_objective")]:
            src = [int(df.cluster_id.iloc[int(s["owner_index"])])
                   for fold in cj["fold_results"]
                   for s in fold[key]["sentences"]]
            counts = {c: src.count(c) for c in clusters}
            total = len(src)
            sel[f"{model}_{name}"] = {
                "n_selections": total,
                "counts": {str(c): counts[c] for c in clusters},
                "shares": {str(c): round(counts[c] / total, 4)
                           for c in clusters}}

    return {
        "description": ("Nearest-sentence assignment "
                        "(text-embedding-3-large) and source clusters of "
                        "cross-fitted benchmark selections. Counts and "
                        "shares are zero-filled over all clusters."),
        "per_sentence_assignment_openai": per_sentence,
        "worst_cluster": {
            "cluster_id": worst, "n": int(wm.sum()),
            "mean_nearest_sentence_similarity":
                round(float(cov[wm].mean()), 4),
            "per_sentence": worst_per_sentence},
        "benchmark_selection_source_clusters": sel,
        "cluster_base_rates": {
            str(c): round(float((df.cluster_id == c).mean()), 4)
            for c in clusters},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true",
                    help="(re)create the artifacts before verifying")
    args = ap.parse_args()

    for topic in WORST:
        path = ANALYSIS / topic / "sentence_assignment.json"
        fresh = compute(topic)
        if args.generate or not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as handle:
                json.dump(fresh, handle, indent=1)
            print(f"{topic}: artifact generated")
        with path.open() as handle:
            saved = json.load(handle)
        require(saved == fresh,
                f"{topic}: sentence_assignment.json does not match a "
                f"recomputation from the analysis-ready inputs")
        # explicit headline assertions used in the manuscript
        if topic == "education":
            for run, rec in saved["benchmark_selection_source_clusters"].items():
                require(rec["counts"]["8"] == 0,
                        f"education {run}: expected zero C8 selections")
        print(f"{topic}: sentence-assignment artifact verified")
    print("PASS")


if __name__ == "__main__":
    main()
