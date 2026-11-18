#!/usr/bin/env python3
"""Aggregate coverage results -> RESULTS.md + CSVs under analyses/llm_summaries/results/.

  analyses/.venv/bin/python analyses/llm_summaries/analyze.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, kendalltau, spearmanr  # noqa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

RES = HERE / "results"
B_BOOT, N_PERM = 2000, 5000


def gini(v):
    v = np.sort(np.asarray(v, float)); n = len(v)
    if n < 2 or v.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * v) / (n * v.sum()))


def bottom_decile_mean(v):
    k = max(1, int(math.ceil(0.10 * len(v))))
    return float(np.partition(v, k - 1)[:k].mean())


def metrics(v, thr):
    return dict(mean=float(v.mean()), bottom_decile=bottom_decile_mean(v), low_rate=float(np.mean(v < thr)),
                low_rate_relative=float(np.mean(v < v.mean() - v.std())), gini=gini(v))


def official_threshold(topic, emb, off):
    """Frozen threshold stored by the original pipeline (benchmark_<emb>.json), else mean-1SD of official."""
    f = ANALYSIS_ROOT[CASES[topic]] / topic / f"benchmark_{emb}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if "frozen_exclusion_threshold" in d:
            return float(d["frozen_exclusion_threshold"]), "benchmark json"
    return float(off.mean() - off.std()), "recomputed mean-1SD"


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None or (isinstance(s, float) and np.isnan(s)) else f"{m:.{d}f} ± {s:.{d}f}"


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def kendalls_w(mat):
    """mat: (m raters, k items) scores -> W without tie correction, and permutation p."""
    m, k = mat.shape
    ranks = np.apply_along_axis(lambda x: pd.Series(x).rank().values, 1, mat)
    def w(r):
        s = ((r.sum(0) - r.sum(0).mean()) ** 2).sum()
        return 12 * s / (m ** 2 * (k ** 3 - k))
    obs = w(ranks)
    rng = np.random.default_rng(0)
    perm = [w(np.stack([rng.permutation(r) for r in ranks])) for _ in range(N_PERM)]
    return float(obs), float((1 + np.sum(np.array(perm) >= obs)) / (N_PERM + 1))


def analyse_topic(topic, df, md, tables):
    rng = np.random.default_rng(20260101)
    case_root = ANALYSIS_ROOT[CASES[topic]] / topic
    clustered = None
    if (case_root / "clustered.parquet").exists():
        clustered = pd.read_parquet(case_root / "clustered.parquet")
    for emb in EMB_MODELS:
        d = df[df.embedding_model == emb]
        if d.empty:
            continue
        ids = d[d.summarizer == "official"].respondent_id.astype(str).tolist()
        n = len(ids)
        series = {}   # (summarizer, seed) -> vector
        for (s, sd), g in d.groupby(["summarizer", "seed"], dropna=False):
            g = g.set_index(g.respondent_id.astype(str)).coverage.reindex(ids)
            series[(s, "" if pd.isna(sd) else str(sd).replace(".0", ""))] = g.values.astype(float)
        off = series[("official", "")]
        thr, src = official_threshold(topic, emb, off)
        summarizers = ["official"] + [s for s in SLUGS if any(k[0] == s for k in series)]
        md.append(f"\n### {topic} — embedding `{emb}`  (n={n}; low-coverage threshold {thr:.4f}, {src})\n")

        # --- A. main metrics
        rows, per = [], {}
        for s in summarizers:
            ms = [metrics(v, thr) for k, v in series.items() if k[0] == s]
            per[s] = ms
            row = dict(summarizer=s, n_seeds=len(ms))
            for key, lab in [("mean", "mean_cov"), ("bottom_decile", "bottom_decile"), ("low_rate", "low_rate"),
                             ("low_rate_relative", "low_rate_own_mean-1SD"), ("gini", "gini")]:
                a = np.array([m[key] for m in ms])
                row[lab] = fmt(a.mean(), a.std(ddof=1) if len(a) > 1 else None, 3)
            rows.append(row)
        t = pd.DataFrame(rows); t.insert(0, "embedding", emb); t.insert(0, "topic", topic)
        tables.setdefault("metrics", []).append(t)
        md.append("Coverage metrics (mean ± sd across seeds; low_rate uses the official frozen threshold):\n")
        md.append(md_table(t.drop(columns=["topic", "embedding"])))

        # --- B. paired comparison vs official (bootstrap over participants)
        rows = []
        for s in summarizers[1:]:
            vecs = [v for k, v in series.items() if k[0] == s]
            idx = rng.integers(0, n, size=(B_BOOT, n))
            def stat(sel, f):
                return np.mean([f(v[sel]) for v in vecs])
            fs = dict(mean=lambda v: v.mean(), bottom_decile=bottom_decile_mean, low_rate=lambda v: np.mean(v < thr))
            row = dict(summarizer=s)
            for key, f in fs.items():
                base = np.mean([f(v) for v in vecs]) - f(off)
                boots = np.array([np.mean([f(v[b]) for v in vecs]) - f(off[b]) for b in idx])
                lo, hi = np.percentile(boots, [2.5, 97.5])
                row[f"Δ{key} (model−official)"] = f"{base:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            rows.append(row)
        t = pd.DataFrame(rows); t.insert(0, "embedding", emb); t.insert(0, "topic", topic)
        tables.setdefault("paired_vs_official", []).append(t)
        md.append("\nPaired difference vs official (seed-averaged; 95% participant-bootstrap CI, B=" + str(B_BOOT) + "):\n")
        md.append(md_table(t.drop(columns=["topic", "embedding"])))

        # --- C. structural: per-summarizer seed-mean vectors
        mean_vec = {s: np.mean([v for k, v in series.items() if k[0] == s], axis=0) for s in summarizers}
        rho = pd.DataFrame(index=summarizers, columns=summarizers, dtype=float)
        for a, b in itertools.product(summarizers, summarizers):
            rho.loc[a, b] = 1.0 if a == b else spearmanr(mean_vec[a], mean_vec[b])[0]
        tables.setdefault("spearman", []).append(rho.round(4).reset_index().rename(columns={"index": "summarizer"})
                                                 .assign(topic=topic, embedding=emb))
        md.append("\nSpearman correlation of participant coverage between summarizers (seed-averaged coverage):\n")
        md.append(md_table(rho.round(3).reset_index().rename(columns={"index": ""})))
        # seed-to-seed stability within each model
        stab = []
        for s in summarizers[1:]:
            ks = [v for k, v in series.items() if k[0] == s]
            if len(ks) > 1:
                stab.append(f"{s}: {np.mean([spearmanr(a, b)[0] for a, b in itertools.combinations(ks, 2)]):.3f}")
        if stab:
            md.append("\nWithin-model seed-to-seed Spearman: " + "; ".join(stab) + "\n")

        # Jaccard of bottom-decile sets vs permutation baseline
        k = max(1, int(math.ceil(0.10 * n)))
        sets = {s: set(np.argsort(mean_vec[s])[:k]) for s in summarizers}
        rows = []
        for a, b in itertools.combinations(summarizers, 2):
            j = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            base = []
            for _ in range(2000):
                x, y = set(rng.choice(n, k, replace=False)), set(rng.choice(n, k, replace=False))
                base.append(len(x & y) / len(x | y))
            base = np.array(base)
            rows.append(dict(pair=f"{a} vs {b}", jaccard=f"{j:.3f}", baseline_mean=f"{base.mean():.3f}",
                             baseline_p95=f"{np.percentile(base, 95):.3f}",
                             perm_p=f"{(1 + np.sum(base >= j)) / (len(base) + 1):.4f}"))
        t = pd.DataFrame(rows); t.insert(0, "embedding", emb); t.insert(0, "topic", topic)
        tables.setdefault("jaccard_bottom_decile", []).append(t)
        md.append(f"\nJaccard overlap of bottom-decile participant sets (k={k}) vs random-set baseline:\n")
        md.append(md_table(t.drop(columns=["topic", "embedding"])))

        # --- D. region (topic-cluster) ranks and Kendall's W
        if clustered is not None:
            try:
                idc = pick_col(clustered.columns, ID_CANDIDATES, what="id column of clustered.parquet")
                cl = clustered.set_index(clustered[idc].astype(str))
                cl = cl.reindex(ids)
            except KeyError:
                cl = clustered if len(clustered) == n else None
            ccol = next((c for c in ["cluster_id", "cluster", "region", "label"] if cl is not None and c in cl.columns), None)
            if cl is not None and ccol:
                lab = cl[ccol].values
                regions = [r for r in sorted(pd.unique(lab[~pd.isna(lab)]))]
                cm = pd.DataFrame({s: [mean_vec[s][lab == r].mean() for r in regions] for s in summarizers},
                                  index=[f"{ccol}={r}" for r in regions])
                cm.insert(0, "n", [int(np.sum(lab == r)) for r in regions])
                tables.setdefault("region_mean_coverage", []).append(cm.round(4).reset_index().rename(columns={"index": "region"})
                                                                      .assign(topic=topic, embedding=emb))
                rk = cm[summarizers].rank()
                md.append(f"\nPer-region ({ccol}, {len(regions)} regions) mean coverage rank (1 = lowest):\n")
                md.append(md_table(rk.astype(int).reset_index().rename(columns={"index": "region"})))
                w_all, p_all = kendalls_w(cm[summarizers].values.T)
                models_only = [s for s in summarizers if s != "official"]
                w_m, p_m = kendalls_w(cm[models_only].values.T) if len(models_only) > 1 else (float("nan"), float("nan"))
                lowest = {s: cm[s].idxmin() for s in summarizers}
                tables.setdefault("kendall_w", []).append(pd.DataFrame([dict(
                    topic=topic, embedding=emb, n_regions=len(regions), W_all=w_all, p_all=p_all,
                    W_models_only=w_m, p_models_only=p_m)]))
                md.append(f"\nKendall's W across summarizers on region coverage: all (incl. official) W={w_all:.3f} (perm p={p_all:.4f}); "
                          f"LLMs only W={w_m:.3f} (p={p_m:.4f}). Lowest-covered region per summarizer: "
                          + ", ".join(f"{s}→{v}" for s, v in lowest.items()) + ".\n")
            else:
                md.append("\n(region analysis skipped: cluster column not found)\n")


def budget_table(topics, md, tables):
    rows = []
    for topic in topics:
        for slug in SLUGS:
            fs = sorted((OUT_DIR / topic / slug).glob("seed_*/summary.json"))
            if not fs:
                continue
            ss = [json.loads(f.read_text()) for f in fs]
            rows.append(dict(topic=topic, model=slug, runs=len(ss), budget=f"{ss[0]['budget_sentences']} sent / ≤{ss[0]['budget_words']} words",
                             sentences=fmt(np.mean([s['n_sentences'] for s in ss]), None, 1),
                             words=f"{np.mean([s['n_words'] for s in ss]):.0f} (range {min(s['n_words'] for s in ss)}–{max(s['n_words'] for s in ss)})",
                             compliant=f"{sum(s['budget_compliant'] for s in ss)}/{len(ss)}",
                             mean_attempts=f"{np.mean([s['attempts'] for s in ss]):.1f}"))
    if rows:
        t = pd.DataFrame(rows)
        tables["budget_compliance"] = [t]
        md.append("## Length and budget compliance\n")
        md.append(md_table(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=ALL_TOPICS)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    md = ["# LLM summariser comparison — results", "",
          "Auto-generated by `analyze.py`. Coverage = max cosine similarity between a participant response and any summary "
          "sentence (original pipeline definition). Official = published summary; models = gpt-4o-mini, qwen3.6-35b-a3b, "
          "glm-4.7-flash under an identical map-reduce protocol (temperature 0.7, seeds 1729-1731).", ""]
    tables, done = {}, []
    budget_table(args.topics, md, tables)
    md.append("\n## Coverage and structure per topic\n")
    for topic in args.topics:
        f = SCORE_DIR / topic / "participant_coverage.csv"
        if not f.exists():
            md.append(f"\n*{topic}: no participant_coverage.csv (run score.py)*\n"); continue
        df = pd.read_csv(f, dtype={"respondent_id": str, "seed": str})
        md.append(f"\n## {topic}")
        analyse_topic(topic, df, md, tables); done.append(topic)
    md.append("\n## How to read the structural analysis\n\n"
              "High agreement shows that the pattern of low coverage is shared across the tested summarizers under this protocol; "
        "it does not separate the contributions of the submissions, the prompt and the summarizer.")
    (HERE / "RESULTS.md").write_text("\n".join(md))
    for name, ts in tables.items():
        pd.concat(ts).to_csv(RES / f"{name}.csv", index=False)
    print(f"wrote RESULTS.md and {len(tables)} CSVs in {RES} (topics: {done})")


if __name__ == "__main__":
    main()
