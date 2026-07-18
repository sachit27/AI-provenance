#!/usr/bin/env python3
"""
06_cross_topic.py — Cross-Topic Replication & Robustness

Tests generalizability and stability of findings:

  1. Cross-topic metric comparison (Education vs Trust)
  2. Cross-topic participant consistency (2,392 overlap respondents)
  3. Multi-model robustness (OpenAI vs sentence-transformers vs Ollama)
  4. Parameter sensitivity (PCA dims, k, exclusion threshold)

References:
  - Efron & Tibshirani (1993), "An Introduction to the Bootstrap"

Usage:
    python 06_cross_topic.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from config import (
    NumpyEncoder, OUTPUT_DIR, TOPICS,
    TRANSPORT_PCA_DIMS, BOOTSTRAP_B,
    ROBUSTNESS_EMBEDDING_MODELS,
)
from analysis_io import core_embeddings, nomic_core_embeddings

warnings.filterwarnings("ignore")


# ─── Helpers ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, stat_fn, B: int = 2000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (estimate, ci_lower, ci_upper)."""
    rng = np.random.RandomState(42)
    n = len(values)
    estimate = float(stat_fn(values))
    boot = np.array([stat_fn(values[rng.randint(0, n, size=n)]) for _ in range(B)])
    ci = (float(np.percentile(boot, 100 * alpha / 2)),
          float(np.percentile(boot, 100 * (1 - alpha / 2))))
    return estimate, ci[0], ci[1]


def gini_coefficient(values: np.ndarray) -> float:
    values = np.sort(values)
    n = len(values)
    if n < 2 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float(np.sum((2 * index - n - 1) * values) / (n * np.sum(values)))


def compute_w2(source, target):
    import ot
    n, m = len(source), len(target)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(source, target, metric="sqeuclidean")
    return float(np.sqrt(ot.emd2(a, b, M)))


# ─── 1. Cross-Topic Metric Comparison ───────────────────────────────────

def compare_metrics() -> dict:
    print(f"\n{'='*60}")
    print(f"1. Cross-Topic Metric Comparison")
    print(f"{'='*60}")

    results = {}
    for topic_name in TOPICS:
        topic_dir = OUTPUT_DIR / topic_name
        transport = json.load(open(topic_dir / "transport.json"))
        topology = json.load(open(topic_dir / "topology.json"))
        results[topic_name] = {
            "wasserstein_2": transport["wasserstein_2"],
            "gini": transport["gini"]["value"],
            "gini_ci": transport["gini"]["ci_95"],
            "exclusion_rate": transport["coverage"]["exclusion_rate"],
            "exclusion_ci": transport["coverage"]["exclusion_ci_95"],
            "mean_coverage": transport["coverage"]["mean"],
            "n_clusters": topology["kmeans"]["best_k"],
            "mean_npmi": topology["topic_coherence"]["mean_npmi"],
            "best_silhouette": topology["kmeans"]["best_silhouette"],
        }

    print(f"\n  {'Metric':<35} {'Education':>12} {'Trust':>12}")
    print(f"  {'-'*59}")
    topics = list(results.keys())
    if len(topics) == 2:
        t1, t2 = topics
        for key in ["wasserstein_2", "gini", "exclusion_rate", "mean_coverage",
                     "n_clusters", "mean_npmi", "best_silhouette"]:
            v1 = results[t1].get(key, "N/A")
            v2 = results[t2].get(key, "N/A")
            v1_str = f"{v1:.4f}" if isinstance(v1, float) else str(v1)
            v2_str = f"{v2:.4f}" if isinstance(v2, float) else str(v2)
            print(f"  {key:<35} {v1_str:>12} {v2_str:>12}")

    return results


# ─── 2. Cross-Topic Participant Consistency ──────────────────────────────

def participant_consistency() -> dict:
    print(f"\n{'='*60}")
    print(f"2. Cross-Topic Participant Consistency")
    print(f"{'='*60}")

    topic_names = list(TOPICS.keys())
    if len(topic_names) < 2:
        print("  Only 1 topic — skipping consistency analysis")
        return {"skipped": True}

    dfs = {}
    for topic_name in topic_names:
        topic_dir = OUTPUT_DIR / topic_name
        df = pd.read_parquet(topic_dir / "clustered.parquet")
        dfs[topic_name] = df

    t1, t2 = topic_names[0], topic_names[1]
    df1, df2 = dfs[t1], dfs[t2]

    # Find overlapping respondents by Internal ID
    if "Internal ID" not in df1.columns or "Internal ID" not in df2.columns:
        # Try matching by id column
        ids1 = set(df1["id"].values) if "id" in df1.columns else set()
        ids2 = set(df2["id"].values) if "id" in df2.columns else set()
        overlap_ids = ids1 & ids2
        id_col = "id"
    else:
        ids1 = set(df1["Internal ID"].values)
        ids2 = set(df2["Internal ID"].values)
        overlap_ids = ids1 & ids2
        id_col = "Internal ID"

    n_overlap = len(overlap_ids)
    print(f"  Overlap respondents: {n_overlap}")

    if n_overlap < 10:
        print("  Too few overlapping respondents for consistency analysis")
        return {"n_overlap": n_overlap, "skipped": True}

    # Merge on overlap
    df1_overlap = df1[df1[id_col].isin(overlap_ids)].set_index(id_col)
    df2_overlap = df2[df2[id_col].isin(overlap_ids)].set_index(id_col)

    common_ids = sorted(df1_overlap.index.intersection(df2_overlap.index))
    df1_overlap = df1_overlap.loc[common_ids]
    df2_overlap = df2_overlap.loc[common_ids]

    result = {"n_overlap": n_overlap}

    # Coverage score correlation across topics
    score_col = "coverage_score" if "coverage_score" in df1_overlap.columns else "coupling_weight"
    if score_col in df1_overlap.columns and score_col in df2_overlap.columns:
        cw1 = df1_overlap[score_col].values
        cw2 = df2_overlap[score_col].values
        rho, p_val = sp_stats.spearmanr(cw1, cw2)
        print(f"  {score_col} Spearman ρ = {rho:.4f} (p={p_val:.4f})")
        result["coverage_spearman_rho"] = round(float(rho), 4)
        result["coverage_spearman_p"] = round(float(p_val), 4)

    # Exclusion consistency across topics
    if "is_excluded" in df1_overlap.columns and "is_excluded" in df2_overlap.columns:
        o1 = df1_overlap["is_excluded"].values.astype(int)
        o2 = df2_overlap["is_excluded"].values.astype(int)

        # 2x2 contingency table
        both = int(((o1 == 1) & (o2 == 1)).sum())
        only1 = int(((o1 == 1) & (o2 == 0)).sum())
        only2 = int(((o1 == 0) & (o2 == 1)).sum())
        neither = int(((o1 == 0) & (o2 == 0)).sum())

        contingency = [[both, only1], [only2, neither]]
        chi2, p_chi, _, _ = sp_stats.chi2_contingency(contingency, correction=False)

        # Tetrachoric approximation via phi coefficient
        phi = (both * neither - only1 * only2) / max(
            np.sqrt(float((both + only1) * (only2 + neither) *
                          (both + only2) * (only1 + neither))), 1e-10
        )

        print(f"  Exclusion contingency table:")
        print(f"    Both excluded: {both}, Only {t1}: {only1}, Only {t2}: {only2}, Neither: {neither}")
        print(f"    χ² = {chi2:.2f}, p = {p_chi:.4f}")
        print(f"    φ = {phi:.4f}")

        result["exclusion_contingency"] = {
            "both": both, f"only_{t1}": only1,
            f"only_{t2}": only2, "neither": neither,
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p_chi), 4),
            "phi": round(float(phi), 4),
        }

    # Register consistency
    if "register" in df1_overlap.columns and "register" in df2_overlap.columns:
        same_register = (df1_overlap["register"].values == df2_overlap["register"].values).mean()
        print(f"  Same register across topics: {same_register:.1%}")
        result["register_agreement"] = round(float(same_register), 4)

    return result


# ─── 3. Multi-Model Robustness ──────────────────────────────────────────

def multi_model_robustness() -> dict:
    print(f"\n{'='*60}")
    print(f"3. Multi-Model Robustness")
    print(f"{'='*60}")

    results = {}

    for topic_name in TOPICS:
        topic_dir = OUTPUT_DIR / topic_name
        print(f"\n  Topic: {topic_name}")

        # Every analysis reads the same versioned participant and summary rows
        # from the single combined embedding set.
        primary_input, primary_summary, _ = core_embeddings(topic_name, "openai")

        df = pd.read_parquet(topic_dir / "clustered.parquet")
        # Use coverage_score (cosine similarity) as primary individual metric
        primary_coverage = df["coverage_score"].values if "coverage_score" in df.columns else None

        topic_results = {}

        for model_name, model_type in ROBUSTNESS_EMBEDDING_MODELS:
            print(f"    {model_name}:")
            if model_name == "all-mpnet-base-v2":
                rob_input, rob_summary, _ = core_embeddings(topic_name, "mpnet")
            elif model_name == "nomic-embed-text":
                rob_input, rob_summary = nomic_core_embeddings(topic_name)
            else:
                print(f"      unsupported robustness model, skipping")
                continue

            # W2 (PCA-reduced)
            pca = PCA(n_components=min(TRANSPORT_PCA_DIMS, rob_input.shape[1]),
                       random_state=42)
            all_embs = np.vstack([rob_input, rob_summary])
            all_pca = pca.fit_transform(all_embs)
            n = len(rob_input)
            source = all_pca[:n]
            target = all_pca[n:]
            w2 = compute_w2(source, target)

            # Coverage score (cosine similarity in original embedding space)
            rob_src = rob_input / (np.linalg.norm(rob_input, axis=1, keepdims=True) + 1e-10)
            rob_tgt = rob_summary / (np.linalg.norm(rob_summary, axis=1, keepdims=True) + 1e-10)
            rob_coverage = (rob_src @ rob_tgt.T).max(axis=1)

            if np.any(rob_coverage < 0):
                raise ValueError("Coverage Gini requires non-negative values")
            gini = gini_coefficient(rob_coverage)
            exclusion_threshold = float(rob_coverage.mean() - rob_coverage.std())
            exclusion_rate = float((rob_coverage < exclusion_threshold).mean())

            print(f"      W2 = {w2:.4f}, Gini = {gini:.4f}, Exclusion rate = {exclusion_rate:.4f}")
            print(f"      Coverage: {rob_coverage.mean():.4f} ± {rob_coverage.std():.4f}")

            # Rank correlation of coverage scores with primary model
            if primary_coverage is not None and len(primary_coverage) == len(rob_coverage):
                rho, p_val = sp_stats.spearmanr(primary_coverage, rob_coverage)
                print(f"      Coverage rank ρ = {rho:.4f} (p={p_val:.6f})")
            else:
                rho, p_val = float("nan"), float("nan")

            # Exclusion agreement with primary model
            if primary_coverage is not None:
                primary_excluded = primary_coverage < (primary_coverage.mean() - primary_coverage.std())
                rob_excluded = rob_coverage < exclusion_threshold
                exclusion_agreement = float((primary_excluded == rob_excluded).mean())
                print(f"      Exclusion agreement: {exclusion_agreement:.1%}")
            else:
                exclusion_agreement = float("nan")

            # Evaluate the primary partition in the alternative embedding
            # space. This does not claim that independently re-clustered labels
            # are identical; it asks whether the same primary groups retain
            # high or low exclusion rates under another coverage model.
            cluster_exclusion = {}
            rob_excluded = rob_coverage < exclusion_threshold
            for cid in sorted(df["cluster_id"].unique()):
                mask = df["cluster_id"].to_numpy() == cid
                cluster_exclusion[str(int(cid))] = {
                    "n": int(mask.sum()),
                    "mean_coverage": float(rob_coverage[mask].mean()),
                    "exclusion_rate": float(rob_excluded[mask].mean()),
                }
            ranked_clusters = sorted(
                cluster_exclusion,
                key=lambda cid: cluster_exclusion[cid]["exclusion_rate"],
                reverse=True,
            )

            topic_results[model_name] = {
                "wasserstein_2": round(w2, 6),
                "gini": round(gini, 4),
                "exclusion_rate": round(exclusion_rate, 4),
                "coverage_mean": round(float(rob_coverage.mean()), 4),
                "coverage_rank_correlation": round(float(rho), 4),
                "coverage_rank_p": round(float(p_val), 6),
                "exclusion_agreement": round(exclusion_agreement, 4),
                "primary_partition_cluster_exclusion": cluster_exclusion,
                "primary_partition_clusters_ranked_most_to_least_excluded": ranked_clusters,
            }

        results[topic_name] = topic_results

    return results


# ─── 4. Parameter Sensitivity ───────────────────────────────────────────

def parameter_sensitivity() -> dict:
    print(f"\n{'='*60}")
    print(f"4. Parameter Sensitivity (Autoresearch-style)")
    print(f"{'='*60}")

    pca_dims_sweep = [20, 50, 100]
    exclusion_sd_sweep = [0.5, 1.0, 1.5, 2.0]  # SD multipliers for exclusion threshold

    results = {}

    for topic_name in TOPICS:
        topic_dir = OUTPUT_DIR / topic_name
        print(f"\n  Topic: {topic_name}")

        input_embs, summary_embs, _ = core_embeddings(topic_name, "openai")
        n = len(input_embs)

        topic_results = {"pca_sweep": [], "exclusion_sweep": [], "k_sweep": []}

        # PCA dimension sweep
        print(f"    PCA dimension sweep: {pca_dims_sweep}")
        for pca_d in pca_dims_sweep:
            d = min(pca_d, input_embs.shape[1])
            pca = PCA(n_components=d, random_state=42)
            all_embs = np.vstack([input_embs, summary_embs])
            all_pca = pca.fit_transform(all_embs)
            source = all_pca[:n]
            target = all_pca[n:]

            w2 = compute_w2(source, target)
            explained = float(pca.explained_variance_ratio_.sum())

            # Gini on cosine coverage scores (not coupling weights)
            src_n = input_embs / (np.linalg.norm(input_embs, axis=1, keepdims=True) + 1e-10)
            tgt_n = summary_embs / (np.linalg.norm(summary_embs, axis=1, keepdims=True) + 1e-10)
            cov = (src_n @ tgt_n.T).max(axis=1)
            if np.any(cov < 0):
                raise ValueError("Coverage Gini requires non-negative values")
            gini = gini_coefficient(cov)

            print(f"      d={d}: W2={w2:.4f}, Gini={gini:.4f}, explained={explained:.1%}")
            topic_results["pca_sweep"].append({
                "pca_dims": d,
                "wasserstein_2": round(w2, 6),
                "gini": round(gini, 4),
                "explained_variance": round(explained, 4),
            })

        # Exclusion threshold sweep (SD multipliers)
        sd_multipliers = [0.5, 1.0, 1.5, 2.0]
        print(f"    Exclusion threshold sweep (SD multipliers): {sd_multipliers}")
        # Cosine coverage for exclusion threshold sweep
        src_n = input_embs / (np.linalg.norm(input_embs, axis=1, keepdims=True) + 1e-10)
        tgt_n = summary_embs / (np.linalg.norm(summary_embs, axis=1, keepdims=True) + 1e-10)
        coverage = (src_n @ tgt_n.T).max(axis=1)
        cov_mean, cov_std = float(coverage.mean()), float(coverage.std())

        # Vary the exclusion threshold multiplier (how many SDs below mean = excluded)
        for mult in sd_multipliers:
            threshold = cov_mean - mult * cov_std
            exclusion_rate = float((coverage < threshold).mean())
            print(f"      threshold=μ−{mult}σ ({threshold:.4f}): exclusion_rate={exclusion_rate:.4f}")
            topic_results["exclusion_sweep"].append({
                "sd_multiplier": mult,
                "threshold": round(threshold, 6),
                "exclusion_rate": round(exclusion_rate, 4),
            })

        # K sweep (re-cluster and check stability)
        print(f"    K sweep for coupling stability...")
        df = pd.read_parquet(topic_dir / "clustered.parquet")
        best_k = int(df["cluster_id"].max()) + 1
        primary_labels = df["cluster_id"].to_numpy(dtype=int)
        excluded = df["is_excluded"].to_numpy(dtype=bool)

        k_range = [max(2, best_k - 2), best_k, best_k + 2]
        pca_cluster = PCA(n_components=50, random_state=42)
        pca_embs = pca_cluster.fit_transform(input_embs)

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(pca_embs)
            sil = silhouette_score(pca_embs, labels)
            cluster_structure = {}
            for cid in sorted(np.unique(labels)):
                mask = labels == cid
                primary_counts = pd.Series(primary_labels[mask]).value_counts()
                dominant_primary = int(primary_counts.index[0])
                cluster_structure[str(int(cid))] = {
                    "n": int(mask.sum()),
                    "exclusion_rate": float(excluded[mask].mean()),
                    "mean_coverage": float(coverage[mask].mean()),
                    "dominant_primary_cluster": dominant_primary,
                    "dominant_primary_overlap_fraction": float(primary_counts.iloc[0] / mask.sum()),
                }
            ranked = sorted(
                cluster_structure,
                key=lambda cid: cluster_structure[cid]["exclusion_rate"],
                reverse=True,
            )
            ari = float(adjusted_rand_score(primary_labels, labels))
            print(
                f"      k={k}: silhouette={sil:.4f}, ARI vs primary={ari:.4f}, "
                f"most excluded={ranked[0]}"
            )
            topic_results["k_sweep"].append({
                "k": k,
                "silhouette": round(sil, 4),
                "adjusted_rand_vs_primary_partition": round(ari, 4),
                "cluster_structure": cluster_structure,
                "clusters_ranked_most_to_least_excluded": ranked,
            })

        results[topic_name] = topic_results

    return results


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    cross_dir = OUTPUT_DIR / "cross_topic"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metric comparison
    metrics = compare_metrics()

    # 2. Participant consistency
    consistency = participant_consistency()

    # 3. Multi-model robustness
    robustness = multi_model_robustness()

    # 4. Parameter sensitivity
    sensitivity = parameter_sensitivity()

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "metric_comparison": metrics,
        "participant_consistency": consistency,
        "multi_model_robustness": robustness,
        "parameter_sensitivity": sensitivity,
    }

    with open(cross_dir / "comparison.json", "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n✓ Cross-topic analysis saved → {cross_dir}/")


if __name__ == "__main__":
    main()
