#!/usr/bin/env python3
"""
04_transport.py — Representational Coverage Analysis

Models the government summary as a selective compression of the public input
distribution and measures representational equity at two levels:

  MACRO (distribution level):
    1. Wasserstein-2 distance between participant distribution P and summary Q
    2. Optimal transport plan (coupling matrix Γ) — the provenance map
    3. Two descriptive geometric references: six-centroid and six-quote extractive
    4. Sentence mass — which summary sentences do the most representational work

  MICRO (individual level):
    5. Cosine coverage score — how semantically close is each participant to
       the nearest summary sentence (in original embedding space)
    6. Excluded voices — participants with coverage below mean - 1 SD
    7. Per-cluster coverage statistics — which topic clusters are least covered
    8. Gini coefficient on coverage scores (inequality of representation)
    9. Permutation test: does cluster membership predict coverage? (F-statistic)
   10. Bootstrap CIs on all key statistics

NOTE on design: The transport plan (gamma) drives W2 and provenance mapping.
Coverage scores use cosine similarity in original embedding space because:
  - the coupling answers a distribution-level allocation question, whereas
    participant-level auditing requires a directly interpretable proximity score;
  - cosine similarity captures semantic proximity to the nearest summary sentence.

References:
  - Villani (2009), "Optimal Transport: Old and New"
  - Peyré & Cuturi (2019), "Computational Optimal Transport"
  - Heusel et al. (2017), "GANs Trained by a Two Time-Scale Update Rule"

Usage:
    python 04_transport.py
"""
import json

import numpy as np
import pandas as pd
import ot
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    NumpyEncoder,
    OUTPUT_DIR, TOPICS,
    TRANSPORT_PCA_DIMS,
    BOOTSTRAP_B, PERMUTATION_N,
)
from analysis_io import core_embeddings


# ─── Core Metrics ──────────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n < 2 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_v = np.sort(np.asarray(values, dtype=float))
    cum = np.cumsum(sorted_v)
    cum = cum / cum[-1] if cum[-1] > 0 else cum
    x = np.linspace(0, 1, len(cum) + 1)
    y = np.concatenate([[0], cum])
    return x, y


# ─── Optimal Transport ────────────────────────────────────────────────────

def compute_transport(source: np.ndarray, target: np.ndarray):
    """
    Exact Earth Mover's Distance + transport plan.
    source: (n, d), target: (m, d) in PCA-reduced space.
    Returns: W2 distance, coupling matrix Γ (n, m).
    """
    n, m = len(source), len(target)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(source, target, metric="sqeuclidean")
    gamma = ot.emd(a, b, M)
    w2 = float(np.sqrt(np.sum(gamma * M)))
    return w2, gamma


def compute_w2_only(source: np.ndarray, target: np.ndarray) -> float:
    n, m = len(source), len(target)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(source, target, metric="sqeuclidean")
    return float(np.sqrt(ot.emd2(a, b, M)))


# ─── Coverage Score (individual representation) ──────────────────────────

def compute_coverage(input_embeddings: np.ndarray,
                     summary_embeddings: np.ndarray) -> np.ndarray:
    """
    Cosine coverage: for each participant, max cosine similarity to any
    summary sentence. Range [−1, 1], higher = better represented.
    Operates in original (high-dimensional) embedding space.
    """
    # Normalize rows for cosine similarity
    src = input_embeddings / (np.linalg.norm(input_embeddings, axis=1, keepdims=True) + 1e-10)
    tgt = summary_embeddings / (np.linalg.norm(summary_embeddings, axis=1, keepdims=True) + 1e-10)
    sim_matrix = src @ tgt.T          # (n, m)
    return sim_matrix.max(axis=1)     # (n,) best-matching similarity per participant


def coverage_exclusion_threshold(coverage: np.ndarray) -> float:
    """
    Exclusion threshold: mean - 1 SD.
    Participants more than 1 SD below mean coverage are classified as
    under-represented (excluded from the summary's semantic scope).
    More interpretable than a fixed percentile when distribution is skewed.
    """
    return float(coverage.mean() - coverage.std())


# ─── Main ──────────────────────────────────────────────────────────────────

def process_topic(topic_name: str):
    topic_dir = OUTPUT_DIR / topic_name
    print(f"\n{'='*60}")
    print(f"Optimal Transport: {topic_name}")
    print(f"{'='*60}")

    df = pd.read_parquet(topic_dir / "clustered.parquet")
    # Archived cluster files may contain downstream columns from superseded
    # causal, entailment, or repair experiments. They are not frozen inputs and
    # must not survive into an authoritative rerun.
    stale_columns = [
        "primary_sentence", "coupling_weight", "coverage_score", "is_excluded",
        "word_count_quintile", "isolation_quintile", "is_short", "is_isolated",
        "is_assertive", "is_hedged", "short_x_isolated", "short_x_hedged",
        "isolated_x_hedged", "triple", "entailment_score", "is_entailed",
        "coverage_regime",
    ]
    df = df.drop(columns=[c for c in stale_columns if c in df.columns])
    input_embeddings, summary_embeddings, summary_sentences = core_embeddings(topic_name)

    n = len(df)
    n_summary = len(summary_embeddings)
    print(f"  Participants: {n}, Summary sentences: {n_summary}")

    # ── PCA for transport (macro level) ──────────────────────────────────
    pca = PCA(n_components=TRANSPORT_PCA_DIMS, random_state=42)
    all_embs = np.vstack([input_embeddings, summary_embeddings])
    all_pca = pca.fit_transform(all_embs)
    source = all_pca[:n]
    target = all_pca[n:]
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA → {TRANSPORT_PCA_DIMS}D (explained: {explained:.1%})")

    # ── Wasserstein-2 + Transport Plan ───────────────────────────────────
    print(f"\n  [MACRO] Wasserstein-2 distance + representational mapping...")
    w2, gamma = compute_transport(source, target)
    print(f"    W2 = {w2:.6f}")

    # Provenance mapping: which summary sentence is nearest to each participant
    df["primary_sentence"] = gamma.argmax(axis=1)
    # Coupling weight retained for provenance visualisation only (not inequality measure)
    df["coupling_weight"] = gamma.max(axis=1) * n

    # Sentence coverage mass: share of participant mass each summary sentence absorbs
    sentence_mass = gamma.sum(axis=0) * n_summary   # normalized so they sum to n_summary
    print(f"    Sentence coverage mass (normalized): {np.round(sentence_mass, 3)}")

    # ── Cosine Coverage Score (micro level) ──────────────────────────────
    print(f"\n  [MICRO] Cosine coverage scores...")
    coverage = compute_coverage(input_embeddings, summary_embeddings)
    df["coverage_score"] = coverage

    mu_cov = float(coverage.mean())
    sd_cov = float(coverage.std())
    exclusion_threshold = coverage_exclusion_threshold(coverage)
    df["is_excluded"] = coverage < exclusion_threshold
    n_excluded = int(df["is_excluded"].sum())

    print(f"    Mean coverage: {mu_cov:.4f} ± {sd_cov:.4f}")
    print(f"    Exclusion threshold (μ−σ): {exclusion_threshold:.4f}")
    print(f"    Excluded voices: {n_excluded} ({100*n_excluded/n:.1f}%)")

    # ── Per-cluster coverage + Wasserstein ───────────────────────────────
    print(f"\n  Per-cluster statistics...")
    cluster_stats = {}
    for cid in sorted(df["cluster_id"].unique()):
        mask = (df["cluster_id"] == cid).values
        cluster_source = source[mask]
        cov_cluster = coverage[mask]
        if len(cluster_source) < 2:
            continue
        cw2 = compute_w2_only(cluster_source, target)
        cluster_stats[cid] = {
            "w2": cw2,
            "mean_coverage": float(cov_cluster.mean()),
            "std_coverage": float(cov_cluster.std()),
            "exclusion_rate": float(df.loc[mask, "is_excluded"].mean()),
            "n": int(mask.sum()),
        }
        print(f"    C{cid} (n={mask.sum():4d}): W2={cw2:.4f}, "
              f"coverage={cov_cluster.mean():.4f}±{cov_cluster.std():.4f}, "
              f"excluded={100*df.loc[mask, 'is_excluded'].mean():.1f}%")

    # ── Gini on coverage scores ───────────────────────────────────────────
    # Use raw non-negative coverage. Gini is translation-sensitive, so shifting
    # by the sample minimum would not match the stated statistic.
    if np.any(coverage < 0):
        raise ValueError("Coverage Gini requires non-negative values")
    gini = gini_coefficient(coverage)
    lorenz_x, lorenz_y = lorenz_curve(coverage)
    print(f"\n  Representational Gini (on coverage): {gini:.4f}")

    # Random-summary inference is intentionally not performed here. Sampling
    # whole participant records creates summaries that are many times longer
    # than the official summary and is therefore not a valid chance benchmark.
    # Exact-length random and same-budget optimized extractive comparisons are
    # implemented in 02_prepare_benchmarks.py and confirmed out of sample
    # in 07_crossfit_benchmarks.py.
    rng = np.random.RandomState(42)
    print(f"\n  Descriptive W2 references...")
    from sklearn.cluster import KMeans
    km_baseline = KMeans(n_clusters=n_summary, random_state=42, n_init=10)
    km_baseline.fit(source)
    centroid_w2 = compute_w2_only(source, km_baseline.cluster_centers_)
    print(f"    Six-centroid W2: {centroid_w2:.4f}")

    # Optimal extractive
    selected = []
    remaining = list(range(n))
    for _ in range(n_summary):
        best_i, best_w2_val = -1, float("inf")
        candidates = rng.choice(remaining, size=min(200, len(remaining)), replace=False)
        for i in candidates:
            trial = source[selected + [i]] if selected else source[[i]]
            tw2 = compute_w2_only(source, trial)
            if tw2 < best_w2_val:
                best_w2_val = tw2
                best_i = i
        selected.append(best_i)
        if best_i in remaining:
            remaining.remove(best_i)
    extractive_w2 = compute_w2_only(source, source[selected])
    print(f"    Greedy six-quote extractive W2: {extractive_w2:.4f}")

    # ── Bootstrap CIs ───────────────────────────────────────────────────
    print(f"\n  Bootstrap CIs (B={BOOTSTRAP_B})...")
    boot_gini = np.zeros(BOOTSTRAP_B)
    boot_cov = np.zeros(BOOTSTRAP_B)
    boot_excluded = np.zeros(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        idx = rng.randint(0, n, size=n)
        cov_b = coverage[idx]
        boot_gini[b] = gini_coefficient(cov_b)
        boot_cov[b] = coverage[idx].mean()
        boot_excluded[b] = (coverage[idx] < exclusion_threshold).mean()

    gini_ci = (float(np.percentile(boot_gini, 2.5)), float(np.percentile(boot_gini, 97.5)))
    cov_ci = (float(np.percentile(boot_cov, 2.5)), float(np.percentile(boot_cov, 97.5)))
    exclusion_ci = (float(np.percentile(boot_excluded, 2.5)), float(np.percentile(boot_excluded, 97.5)))
    print(f"    Gini: {gini:.4f} [{gini_ci[0]:.4f}, {gini_ci[1]:.4f}]")
    print(f"    Mean coverage: {mu_cov:.4f} [{cov_ci[0]:.4f}, {cov_ci[1]:.4f}]")
    print(f"    Exclusion rate: {n_excluded/n:.4f} [{exclusion_ci[0]:.4f}, {exclusion_ci[1]:.4f}]")

    # ── Permutation test: does cluster membership predict coverage? ───────
    # H0: cluster labels are unrelated to coverage scores
    # Test: one-way ANOVA F-statistic vs. permutation null
    print(f"  Permutation test: cluster→coverage F-statistic (N={PERMUTATION_N})...")
    cluster_labels = df["cluster_id"].values
    groups = [coverage[cluster_labels == cid]
               for cid in np.unique(cluster_labels)]
    observed_F, _ = sp_stats.f_oneway(*groups)
    observed_F = float(observed_F)

    perm_F = np.zeros(PERMUTATION_N)
    for p in range(PERMUTATION_N):
        shuffled_labels = rng.permutation(cluster_labels)
        groups_perm = [coverage[shuffled_labels == cid]
                        for cid in np.unique(cluster_labels)]
        perm_F[p], _ = sp_stats.f_oneway(*groups_perm)

    # +1 correction (Phipson & Smyth, 2010): empirical p-values from random
    # permutations should never be exactly zero.
    p_value_F = float((1 + (perm_F >= observed_F).sum()) / (PERMUTATION_N + 1))
    print(f"    Observed F = {observed_F:.4f}, p = {p_value_F:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    df.to_parquet(topic_dir / "clustered.parquet", index=False)
    np.save(topic_dir / "coupling_matrix.npy", gamma)

    transport = {
        "wasserstein_2": w2,
        "pca_dims": TRANSPORT_PCA_DIMS,
        "pca_explained_variance": float(explained),
        "n_participants": n,
        "n_summary_sentences": n_summary,
        "coverage": {
            "mean": round(mu_cov, 6),
            "std": round(sd_cov, 6),
            "ci_95": [round(cov_ci[0], 6), round(cov_ci[1], 6)],
            "exclusion_threshold": round(exclusion_threshold, 6),
            "excluded_count": n_excluded,
            "exclusion_rate": round(n_excluded / n, 4),
            "exclusion_ci_95": [round(exclusion_ci[0], 4), round(exclusion_ci[1], 4)],
        },
        "gini": {
            "value": round(gini, 6),
            "ci_95": [round(gini_ci[0], 6), round(gini_ci[1], 6)],
            "cluster_F_statistic": round(observed_F, 4),
            "cluster_F_p_value": round(p_value_F, 4),
        },
        "per_cluster": {
            str(cid): {
                "w2": round(v["w2"], 6),
                "mean_coverage": round(v["mean_coverage"], 6),
                "std_coverage": round(v["std_coverage"], 6),
                "exclusion_rate": round(v["exclusion_rate"], 4),
                "n": v["n"],
            }
            for cid, v in cluster_stats.items()
        },
        "baselines": {
            "six_centroid_w2": round(centroid_w2, 6),
            "six_quote_extractive_w2": round(extractive_w2, 6),
            "actual_w2": round(w2, 6),
            "interpretation": (
                "Descriptive geometric references only; chance and feasible-"
                "frontier inference is reported by the exact-length and "
                "cross-fitted budget-matched benchmark analyses."
            ),
        },
        "lorenz_curve": {
            "x": lorenz_x.tolist(),
            "y": lorenz_y.tolist(),
        },
        "sentence_mass": {
            str(j): {
                "sentence": summary_sentences[j] if j < len(summary_sentences) else "",
                "mass": round(float(sentence_mass[j]), 6),
                "nearest_participant_similarity": round(float(
                    cosine_similarity(
                        summary_embeddings[[j]],
                        input_embeddings,
                    ).max()
                ), 6),
            }
            for j in range(n_summary)
        },
    }

    with open(topic_dir / "transport.json", "w") as f:
        json.dump(transport, f, indent=2, cls=NumpyEncoder)

    print(f"\n  ✓ Saved transport → {topic_dir}/")


def main():
    for topic_name in TOPICS:
        process_topic(topic_name)
    print(f"\n✓ Representational coverage analysis complete for all topics.")


if __name__ == "__main__":
    main()
