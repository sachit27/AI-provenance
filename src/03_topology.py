#!/usr/bin/env python3
"""
03_topology.py — Semantic Structure Discovery

Discovers topic structure via:
  1. HDBSCAN (density-based, reports natural cluster count + noise)
  2. K-Means with multi-metric ensemble for optimal k:
     - Silhouette score (Rousseeuw, 1987)
     - Calinski-Harabasz index (Calinski & Harabasz, 1974)
     - Davies-Bouldin index (Davies & Bouldin, 1979)
     - Gap statistic (Tibshirani et al., 2001)
     k chosen by majority vote across metrics
  3. Bootstrap stability validation (Adjusted Rand Index)
  4. Semantic cluster labeling:
     - LLM-based: GPT-4o-mini reads representative documents → 3-5 word label
     - KeyBERT backup: contextual keyphrase extraction (reproducible)
  5. NPMI topic coherence validation (Bouma, 2009)
  6. Soft cluster membership
  7. UMAP 2D projection

References:
  - Rousseeuw (1987), "Silhouettes: a graphical aid"
  - Calinski & Harabasz (1974), "A dendrite method for cluster analysis"
  - Davies & Bouldin (1979), "A cluster separation measure"
  - Tibshirani et al. (2001), "Estimating the number of clusters via the gap statistic"
  - McInnes et al. (2017), "hdbscan: Hierarchical density based clustering"
  - Grootendorst (2020), "KeyBERT: Minimal keyword extraction with BERT"
  - Bouma (2009), "Normalized PMI in collocation extraction"

Usage:
    python 03_topology.py
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score, adjusted_rand_score,
)
from scipy.spatial.distance import cdist
import hdbscan
import umap

from config import (
    NumpyEncoder,
    OUTPUT_DIR, TOPICS, OPENAI_API_KEY,
    PCA_DIMS, HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES,
    KMEANS_K_RANGE, CLUSTER_COLORS,
)
from analysis_io import core_embeddings

warnings.filterwarnings("ignore", category=FutureWarning)

# Number of bootstrap iterations for stability analysis
BOOTSTRAP_ITERS = 100
BOOTSTRAP_SUBSAMPLE = 0.8

# Number of representative docs per cluster for LLM labeling
N_REPRESENTATIVE = 10


# ─── Multi-Metric Optimal k Selection ────────────────────────────────────────

def gap_statistic(X: np.ndarray, k_range: range, n_ref: int = 20,
                  random_state: int = 42) -> dict[int, float]:
    """
    Gap statistic (Tibshirani et al., 2001).
    Compares log(W_k) for the data vs uniform reference data.
    Higher gap = stronger clustering signal vs random.
    """
    rng = np.random.RandomState(random_state)

    def _wk(data, labels):
        """Within-cluster sum of squares."""
        w = 0.0
        for lab in np.unique(labels):
            members = data[labels == lab]
            center = members.mean(axis=0)
            w += np.sum((members - center) ** 2)
        return w

    # Bounding box for reference
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    gap_scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        log_wk = np.log(max(_wk(X, labels), 1e-10))

        # Reference datasets
        ref_log_wks = []
        for _ in range(n_ref):
            ref_data = rng.uniform(mins, maxs, size=X.shape)
            ref_labels = KMeans(n_clusters=k, random_state=42, n_init=5).fit_predict(ref_data)
            ref_log_wks.append(np.log(max(_wk(ref_data, ref_labels), 1e-10)))

        gap = np.mean(ref_log_wks) - log_wk
        gap_scores[k] = gap

    return gap_scores


def select_k_by_consensus(sil: dict, ch: dict, db: dict, gap: dict,
                          k_range: range) -> tuple[int, dict]:
    """
    Each metric votes for its best k. Pick k with most votes.
    Tie-break by silhouette (most widely used).
    Returns (best_k, vote_details).
    """
    votes = {k: 0 for k in k_range}
    metric_picks = {}

    # Silhouette: higher is better
    best_sil = max(sil, key=sil.get)
    votes[best_sil] += 1
    metric_picks["silhouette"] = best_sil

    # Calinski-Harabasz: higher is better
    best_ch = max(ch, key=ch.get)
    votes[best_ch] += 1
    metric_picks["calinski_harabasz"] = best_ch

    # Davies-Bouldin: LOWER is better
    best_db = min(db, key=db.get)
    votes[best_db] += 1
    metric_picks["davies_bouldin"] = best_db

    # Gap statistic: higher is better
    best_gap = max(gap, key=gap.get)
    votes[best_gap] += 1
    metric_picks["gap_statistic"] = best_gap

    # Winner = most votes, tie-break by silhouette
    max_votes = max(votes.values())
    candidates = [k for k, v in votes.items() if v == max_votes]
    if len(candidates) == 1:
        best_k = candidates[0]
    else:
        best_k = max(candidates, key=lambda k: sil.get(k, 0))

    return best_k, {
        "metric_picks": metric_picks,
        "votes": {str(k): v for k, v in votes.items() if v > 0},
        "consensus_k": best_k,
        "consensus_votes": votes[best_k],
        "total_metrics": 4,
    }


def bootstrap_stability(X: np.ndarray, k: int, n_iter: int = BOOTSTRAP_ITERS,
                        subsample_frac: float = BOOTSTRAP_SUBSAMPLE) -> dict:
    """
    Bootstrap stability: subsample 80% of data, cluster, compare to full-data
    assignments via Adjusted Rand Index. Stable k → ARI > 0.8 consistently.
    """
    full_km = KMeans(n_clusters=k, random_state=42, n_init=10)
    full_labels = full_km.fit_predict(X)
    n = len(X)
    n_sub = int(n * subsample_frac)

    rng = np.random.RandomState(42)
    aris = []

    for _ in range(n_iter):
        idx = rng.choice(n, size=n_sub, replace=False)
        sub_X = X[idx]
        sub_km = KMeans(n_clusters=k, random_state=rng.randint(0, 10000), n_init=10)
        sub_labels = sub_km.fit_predict(sub_X)

        # Assign full data to subsample centroids for ARI comparison
        full_to_sub = sub_km.predict(X)
        ari = adjusted_rand_score(full_labels, full_to_sub)
        aris.append(ari)

    aris = np.array(aris)
    return {
        "mean_ari": round(float(aris.mean()), 4),
        "std_ari": round(float(aris.std()), 4),
        "min_ari": round(float(aris.min()), 4),
        "pct_above_0.8": round(float((aris > 0.8).mean()), 4),
        "stable": bool(aris.mean() > 0.8),
    }


# ─── Semantic Cluster Labeling ────────────────────────────────────────────────

def get_representative_docs(texts: list[str], embeddings: np.ndarray,
                            labels: np.ndarray, centroids: np.ndarray,
                            n_repr: int = N_REPRESENTATIVE) -> dict[int, list[str]]:
    """For each cluster, select the n_repr documents nearest to the centroid."""
    repr_docs = {}
    for cid in range(len(centroids)):
        mask = labels == cid
        cluster_idx = np.where(mask)[0]
        cluster_embs = embeddings[mask]
        dists = np.linalg.norm(cluster_embs - centroids[cid], axis=1)
        nearest = np.argsort(dists)[:n_repr]
        repr_docs[cid] = [texts[cluster_idx[i]] for i in nearest]
    return repr_docs


def llm_label_clusters(repr_docs: dict[int, list[str]], topic_name: str) -> dict[int, str]:
    """
    Send representative documents to GPT-4o-mini to generate a concise
    thematic label for each cluster. Falls back to KeyBERT on API failure.
    """
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    llm_labels = {}

    for cid, docs in repr_docs.items():
        docs_text = "\n---\n".join(f"[{i+1}] {d[:300]}" for i, d in enumerate(docs))
        prompt = (
            f"Below are {len(docs)} representative responses from a Canadian government "
            f"public consultation about '{topic_name}'. They were grouped together by "
            f"semantic similarity.\n\n"
            f"{docs_text}\n\n"
            f"What is the specific theme these responses share? "
            f"Give a concise label (3-6 words, no articles). "
            f"Be specific — avoid generic labels like 'AI concerns' or 'public opinion'. "
            f"Focus on what distinguishes this group from other groups about the same topic.\n\n"
            f"Label:"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=30,
                temperature=0.1,
            )
            label = response.choices[0].message.content.strip().strip('"\'.')
            llm_labels[cid] = label
        except Exception as e:
            print(f"      WARNING: LLM labeling failed for C{cid}: {e}")
            llm_labels[cid] = f"Cluster {cid}"

    return llm_labels


def keybert_label_clusters(repr_docs: dict[int, list[str]],
                           n_keywords: int = 10) -> dict[int, list[str]]:
    """
    KeyBERT keyphrase extraction from representative documents.
    Uses contextual embeddings (not bag-of-words).
    """
    from keybert import KeyBERT

    kw_model = KeyBERT()
    keybert_labels = {}

    for cid, docs in repr_docs.items():
        combined_text = " ".join(docs)
        keywords = kw_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=n_keywords,
            use_mmr=True,         # Maximal Marginal Relevance for diversity
            diversity=0.5,
        )
        keybert_labels[cid] = [kw for kw, _ in keywords]

    return keybert_labels


# ─── NPMI Topic Coherence (Bouma, 2009) ───────────────────────────────────────

def compute_npmi(documents: list[str], topic_terms: list[str],
                 top_n: int = 10) -> float:
    """
    Normalized Pointwise Mutual Information for topic coherence.
    Higher = more coherent. Range: [-1, 1]. >0 is considered coherent.
    """
    terms = topic_terms[:top_n]
    if len(terms) < 2:
        return 0.0

    n_docs = len(documents)
    term_doc_freq = {}
    for term in terms:
        count = sum(1 for doc in documents if term.lower() in doc.lower())
        term_doc_freq[term] = count

    npmi_sum = 0.0
    n_pairs = 0
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            t1, t2 = terms[i], terms[j]
            p1 = term_doc_freq[t1] / n_docs
            p2 = term_doc_freq[t2] / n_docs
            if p1 == 0 or p2 == 0:
                continue

            co_occur = sum(
                1 for doc in documents
                if t1.lower() in doc.lower() and t2.lower() in doc.lower()
            )
            p12 = co_occur / n_docs

            if p12 == 0:
                npmi_sum += -1.0
            else:
                pmi = np.log(p12 / (p1 * p2))
                npmi = pmi / (-np.log(p12))
                npmi_sum += npmi
            n_pairs += 1

    return float(npmi_sum / max(n_pairs, 1))


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_topic(topic_name: str):
    topic_dir = OUTPUT_DIR / topic_name
    topic_dir.mkdir(parents=True, exist_ok=True)
    # Cluster annotations do not affect fitting or any numerical result.  Reuse
    # the versioned, author-reviewed labels when present so an ordinary rerun
    # never makes a new generative-model call or silently changes terminology.
    previous_topology = None
    if (topic_dir / "topology.json").exists():
        with open(topic_dir / "topology.json") as handle:
            previous_topology = json.load(handle)
    print(f"\n{'='*60}")
    print(f"Topology: {topic_name}")
    print(f"{'='*60}")

    df = pd.read_parquet(topic_dir / "clean.parquet")
    input_embeddings, summary_embeddings, _ = core_embeddings(topic_name)

    n = len(df)
    texts = df["text"].tolist()
    print(f"  Participants: {n}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    pca_embs = pca.fit_transform(input_embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA → {PCA_DIMS}D (explained variance: {explained:.1%})")

    summary_pca = pca.transform(summary_embeddings)

    # ── HDBSCAN ──────────────────────────────────────────────────────────────
    print(f"\n  HDBSCAN (min_cluster={HDBSCAN_MIN_CLUSTER})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
    )
    hdbscan_labels = clusterer.fit_predict(pca_embs)
    n_hdbscan = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    n_noise = int((hdbscan_labels == -1).sum())
    print(f"    Clusters: {n_hdbscan}, Noise: {n_noise} ({100*n_noise/n:.1f}%)")

    df["hdbscan_cluster"] = hdbscan_labels

    # ── Multi-Metric K-Means Selection ───────────────────────────────────────
    print(f"\n  K-Means: multi-metric sweep (k={KMEANS_K_RANGE.start}..{KMEANS_K_RANGE.stop-1})...")

    sil_scores = {}
    ch_scores = {}
    db_scores = {}

    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(pca_embs)
        sil = silhouette_score(pca_embs, labels)
        ch = calinski_harabasz_score(pca_embs, labels)
        db = davies_bouldin_score(pca_embs, labels)
        sil_scores[k] = sil
        ch_scores[k] = ch
        db_scores[k] = db
        print(f"    k={k:2d}  Sil={sil:.4f}  CH={ch:.0f}  DB={db:.3f}")

    print(f"\n  Computing gap statistic (reference datasets)...")
    gap_scores = gap_statistic(pca_embs, KMEANS_K_RANGE)
    for k in KMEANS_K_RANGE:
        print(f"    k={k:2d}  Gap={gap_scores[k]:.4f}")

    best_k, consensus = select_k_by_consensus(
        sil_scores, ch_scores, db_scores, gap_scores, KMEANS_K_RANGE
    )
    print(f"\n  Metric votes:")
    for metric, pick in consensus["metric_picks"].items():
        print(f"    {metric:20s} → k={pick}")
    print(f"  → Consensus k={best_k} ({consensus['consensus_votes']}/{consensus['total_metrics']} votes)")

    # ── Bootstrap Stability — stability-first override ───────────────────────
    # If consensus k is unstable (ARI < 0.8), search downward for the highest
    # stable k. This prevents over-fragmentation on datasets with weak structure.
    MAX_STABLE_K = 10   # cap search; above this clusters are rarely stable
    print(f"\n  Bootstrap stability check (k={best_k}, {BOOTSTRAP_ITERS} iterations)...")
    stability = bootstrap_stability(pca_embs, best_k)
    print(f"    Mean ARI:  {stability['mean_ari']:.3f} ± {stability['std_ari']:.3f}")
    print(f"    Stable:    {'YES' if stability['stable'] else 'NO — searching for stable k'}")

    stability_override = False
    if not stability["stable"]:
        # Search from min(best_k-1, MAX_STABLE_K) downward
        search_ks = sorted(
            [k for k in KMEANS_K_RANGE if k < best_k and k <= MAX_STABLE_K],
            reverse=True
        )
        for candidate_k in search_ks:
            print(f"    Checking k={candidate_k}...")
            cand_stab = bootstrap_stability(pca_embs, candidate_k)
            print(f"      ARI={cand_stab['mean_ari']:.3f}  stable={cand_stab['stable']}")
            if cand_stab["stable"]:
                print(f"    → Overriding k={best_k} → k={candidate_k} (stability criterion)")
                consensus["stability_override"] = {
                    "original_consensus_k": best_k,
                    "stable_k": candidate_k,
                    "reason": f"Consensus k={best_k} ARI={stability['mean_ari']:.3f} < 0.8; "
                              f"k={candidate_k} ARI={cand_stab['mean_ari']:.3f} is stable"
                }
                best_k = candidate_k
                stability = cand_stab
                stability_override = True
                break
        if not stability_override:
            print(f"    WARNING: No stable k found in range; using silhouette-best k")
            best_k = max(sil_scores, key=sil_scores.get)
            stability = bootstrap_stability(pca_embs, best_k)
            consensus["stability_override"] = {
                "original_consensus_k": consensus["consensus_k"],
                "stable_k": best_k,
                "reason": "No stable k found; fell back to silhouette-best k"
            }

    print(f"  → Final k={best_k}  ARI={stability['mean_ari']:.3f}")
    print(f"    % > 0.80:  {stability['pct_above_0.8']:.1%}")

    # Final K-Means with selected k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(pca_embs)
    df["cluster_id"] = kmeans_labels
    centroids = kmeans.cluster_centers_

    # ── Soft cluster membership ──────────────────────────────────────────────
    distances = cdist(pca_embs, centroids, metric="euclidean")
    inv_dist = 1.0 / (distances + 1e-8)
    soft_membership = inv_dist / inv_dist.sum(axis=1, keepdims=True)

    # ── Semantic isolation ───────────────────────────────────────────────────
    df["semantic_isolation"] = 0.0
    for cid in range(best_k):
        mask = df["cluster_id"] == cid
        cluster_embs = pca_embs[mask.values]
        centroid = centroids[cid]
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        df.loc[mask, "semantic_isolation"] = dists

    # ── Representative Documents ─────────────────────────────────────────────
    print(f"\n  Selecting {N_REPRESENTATIVE} representative docs per cluster...")
    repr_docs = get_representative_docs(texts, pca_embs, kmeans_labels, centroids)

    # ── Versioned cluster annotations ────────────────────────────────────────
    saved_labels = (previous_topology or {}).get("topic_labels_llm", {})
    saved_keybert = (previous_topology or {}).get("topic_labels_keybert", {})
    annotation_ids = {str(cid) for cid in range(best_k)}
    if set(saved_labels) == annotation_ids and set(saved_keybert) == annotation_ids:
        print("  Reusing versioned, author-reviewed cluster annotations...")
        llm_labels = {int(cid): label for cid, label in saved_labels.items()}
        keybert_labels = {
            int(cid): terms for cid, terms in saved_keybert.items()
        }
        annotation_source = "versioned author-reviewed annotations"
    else:
        print("  Generating initial GPT-4o-mini cluster labels...")
        llm_labels = llm_label_clusters(repr_docs, topic_name)
        print("  Generating initial KeyBERT backup labels...")
        keybert_labels = keybert_label_clusters(repr_docs)
        annotation_source = "initial GPT-4o-mini and KeyBERT generation"
    for cid in range(best_k):
        n_c = int((kmeans_labels == cid).sum())
        print(f"    C{cid} (n={n_c:4d}): {llm_labels.get(cid, '?')}")

    # ── KeyBERT Backup Labels ────────────────────────────────────────────────
    for cid in range(best_k):
        terms = keybert_labels.get(cid, [])[:5]
        print(f"    C{cid}: {', '.join(terms)}")

    # ── NPMI coherence (using KeyBERT terms) ─────────────────────────────────
    print(f"  Computing NPMI topic coherence...")
    coherence_scores = {}
    for cid, terms in sorted(keybert_labels.items()):
        npmi = compute_npmi(texts, terms[:10])
        coherence_scores[cid] = npmi
        size = int((kmeans_labels == cid).sum())
        print(f"    C{cid} (n={size:4d}, NPMI={npmi:.3f})")

    mean_npmi = np.mean(list(coherence_scores.values()))
    print(f"    Mean NPMI: {mean_npmi:.3f} ({'coherent' if mean_npmi > 0 else 'WARNING: low coherence'})")

    # ── UMAP 2D ──────────────────────────────────────────────────────────────
    print(f"\n  UMAP → 2D...")
    combined = np.vstack([pca_embs, summary_pca])
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    combined_2d = reducer.fit_transform(combined)

    viz_2d = combined_2d[:n]
    summary_2d = combined_2d[n:]
    df["umap_x"] = viz_2d[:, 0]
    df["umap_y"] = viz_2d[:, 1]

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_parquet(topic_dir / "clustered.parquet", index=False)
    np.save(topic_dir / "soft_membership.npy", soft_membership)

    topology = {
        "n_participants": n,
        "pca_dims": PCA_DIMS,
        "pca_explained_variance": float(explained),
        "hdbscan": {
            "n_clusters": n_hdbscan,
            "n_noise": n_noise,
            "noise_pct": round(100 * n_noise / n, 1),
        },
        "kmeans": {
            "best_k": best_k,
            "selection_method": "multi-metric consensus (Sil + CH + DB + Gap)",
            "consensus": consensus,
            "silhouette_scores": {str(k): round(s, 4) for k, s in sil_scores.items()},
            "calinski_harabasz": {str(k): round(s, 1) for k, s in ch_scores.items()},
            "davies_bouldin": {str(k): round(s, 4) for k, s in db_scores.items()},
            "gap_statistic": {str(k): round(s, 4) for k, s in gap_scores.items()},
            "best_silhouette": round(sil_scores[best_k], 4),
            "bootstrap_stability": stability,
        },
        "topic_coherence": {
            "method": "NPMI (Bouma 2009)",
            "per_cluster": {str(k): round(v, 4) for k, v in coherence_scores.items()},
            "mean_npmi": round(mean_npmi, 4),
        },
        "labeling": {
            "primary_method": "Author-reviewed labels initially assisted by GPT-4o-mini on centroid-nearest documents",
            "backup_method": "KeyBERT (MMR-diversified contextual keyphrases)",
            "annotation_source_this_run": annotation_source,
            "n_representative_docs": N_REPRESENTATIVE,
        },
        "cluster_stats": [],
        # LLM labels → used in figures (readable)
        "topic_labels_llm": {str(k): v for k, v in llm_labels.items()},
        # KeyBERT labels → used for NPMI and reproducibility
        "topic_labels_keybert": {str(k): v for k, v in keybert_labels.items()},
        # For backward compatibility, topic_labels uses LLM labels
        "topic_labels": {str(k): v for k, v in llm_labels.items()},
        "summary_2d": summary_2d.tolist(),
    }

    for cid in range(best_k):
        mask = df["cluster_id"] == cid
        cluster_df = df[mask]
        topology["cluster_stats"].append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "label_llm": llm_labels.get(cid, ""),
            "label_keybert": ", ".join(keybert_labels.get(cid, [])[:3]),
            "npmi": round(coherence_scores.get(cid, 0), 4),
            "mean_isolation": round(float(cluster_df["semantic_isolation"].mean()), 4),
        })

    with open(topic_dir / "topology.json", "w") as f:
        json.dump(topology, f, indent=2, cls=NumpyEncoder)

    print(f"\n  ✓ Saved topology → {topic_dir}/")


def main():
    for topic_name in TOPICS:
        process_topic(topic_name)
    print(f"\n✓ Topology complete for all topics.")


if __name__ == "__main__":
    main()
