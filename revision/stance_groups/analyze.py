#!/usr/bin/env python3
"""Stance-group / respondent-type comparison of per-participant summary coverage.

Run from canadian_data root:
    revision/.venv/bin/python revision/stance_groups/analyze.py [--perms 10000] [--boot 2000]

Needs: respondent_groups.csv (build_groups.py), MAPPING.json (frozen a-priori),
official coverage  revision/scotland_original/analysis/<topic>/participant_coverage.csv
optional generated revision/llm_summaries/scores/<topic>/participant_coverage.csv
Dependencies: numpy, pandas (scipy optional: t-distribution p-values; otherwise normal approx).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats as _st
except Exception:  # pragma: no cover
    _st = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20240611
MIN_N = 10


# ----------------------------------------------------------------- statistics
def perm_p(y, n_min, B, rng, obs):
    n = len(y); tot = y.sum(); n_maj = n - n_min
    cnt = 0; done = 0; chunk = 500
    while done < B:
        c = min(chunk, B - done)
        idx = np.argpartition(rng.random((c, n)), n_min - 1, axis=1)[:, :n_min]
        s = y[idx].sum(axis=1)
        d = s / n_min - (tot - s) / n_maj
        cnt += int((np.abs(d) >= abs(obs) - 1e-12).sum()); done += c
    return (cnt + 1) / (B + 1)


def boot_ci(a, b, B, rng):
    ma = a[rng.integers(0, len(a), (B, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), (B, len(b)))].mean(axis=1)
    d = ma - mb
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def ols_hc3(y, minor, wc):
    X = np.column_stack([np.ones(len(y)), minor.astype(float), np.log(np.clip(wc, 1, None))])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    w = (e / (1 - h)) ** 2
    cov = XtXi @ (X.T * w) @ X @ XtXi
    se = math.sqrt(cov[1, 1]); b = beta[1]; df = len(y) - X.shape[1]
    t = b / se if se > 0 else float("nan")
    if _st is not None:
        p = float(2 * _st.t.sf(abs(t), df)); crit = float(_st.t.ppf(0.975, df))
    else:
        p = math.erfc(abs(t) / math.sqrt(2)); crit = 1.96
    return b, se, b - crit * se, b + crit * se, p


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); out = np.full(m, np.nan)
    ok = np.where(~np.isnan(ps))[0]
    order = ok[np.argsort(ps[ok])]; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ok) - rank) * ps[i]); out[i] = min(1.0, run)
    return out


def compare(df, lab_col, min_lab, maj_lab, B_perm, B_boot, seed):
    """df has coverage, word_count, bottom, low_coverage(optional) and a label column."""
    a = df[df[lab_col] == min_lab]; b = df[df[lab_col] == maj_lab]
    sub = pd.concat([a, b])
    y = sub["coverage"].to_numpy(float)
    ya, yb = a["coverage"].to_numpy(float), b["coverage"].to_numpy(float)
    obs = ya.mean() - yb.mean()
    rng = np.random.default_rng(seed)
    lo, hi = boot_ci(ya, yb, B_boot, rng)
    p = perm_p(y, len(a), B_perm, rng, obs)
    coef, se, alo, ahi, ap = ols_hc3(y, np.r_[np.ones(len(a)), np.zeros(len(b))],
                                     sub["word_count"].to_numpy(float))
    r = dict(minority_label=min_lab, majority_label=maj_lab, n_min=len(a), n_maj=len(b),
             mean_min=ya.mean(), mean_maj=yb.mean(), diff=obs, ci_lo=lo, ci_hi=hi, p_perm=p,
             share_bottom10_min=a["bottom"].mean(), share_bottom10_maj=b["bottom"].mean(),
             mean_wc_min=a["word_count"].mean(), mean_wc_maj=b["word_count"].mean(),
             adj_coef=coef, adj_se=se, adj_lo=alo, adj_hi=ahi, adj_p=ap)
    r["bottom10_diff"] = r["share_bottom10_min"] - r["share_bottom10_maj"]
    if "low_coverage" in df and df["low_coverage"].notna().any():
        r["lowcov_rate_min"] = a["low_coverage"].astype(float).mean()
        r["lowcov_rate_maj"] = b["low_coverage"].astype(float).mean()
    return r


# ------------------------------------------------------------------- loading
def load_cov(path, official):
    df = pd.read_csv(path, dtype={"respondent_id": str})
    if "summarizer" not in df: df["summarizer"] = "official"
    if not official and "seed" in df:
        keep = [c for c in ["word_count", "region"] if c in df]
        df = (df.groupby(["respondent_id", "embedding_model", "summarizer"], as_index=False)
                .agg(coverage=("coverage", "mean"), n_seeds=("seed", "nunique")))
    return df


def word_counts(topic, official_df):
    if official_df is not None and "word_count" in official_df:
        return official_df.drop_duplicates("respondent_id").set_index("respondent_id")["word_count"]
    fp = ROOT / f"revision/scotland/datasets/{topic}/records.jsonl"
    d = {}
    if fp.exists():
        for line in fp.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line); d[str(r["respondent_id"])] = r.get("word_count", np.nan)
    return pd.Series(d, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--official-root", default=str(ROOT / "revision/scotland_original/analysis"))
    ap.add_argument("--generated-root", default=str(ROOT / "revision/llm_summaries/scores"))
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    out = Path(a.out)
    mapping = json.loads((HERE / "MAPPING.json").read_text(encoding="utf-8"))
    groups = pd.read_csv(HERE / "respondent_groups.csv", dtype=str).set_index("respondent_id")
    rt = mapping["respondent_type"]

    rows, sizes = [], []
    n_inputs = 0
    for source, root in (("official", Path(a.official_root)), ("generated", Path(a.generated_root))):
        for topic, tinfo in mapping["topics"].items():
            fp = root / topic / "participant_coverage.csv"
            if not fp.exists():
                continue
            n_inputs += 1
            cov_all = load_cov(fp, source == "official")
            off_fp = Path(a.official_root) / topic / "participant_coverage.csv"
            off_df = pd.read_csv(off_fp, dtype={"respondent_id": str}) if off_fp.exists() else None
            wc = word_counts(topic, off_df)
            analyses = [(x["id"], x["kind"], x["column"]) for x in tinfo["analyses"]]
            analyses.append((f"{topic}_resp_type", "secondary_respondent_type", rt["column"]))
            for (emb, summ), blk in cov_all.groupby(["embedding_model", "summarizer"]):
                blk = blk.drop_duplicates("respondent_id").copy()
                blk["word_count"] = blk["respondent_id"].map(wc)
                if blk["word_count"].isna().all(): blk["word_count"] = 1.0
                blk["word_count"] = blk["word_count"].fillna(blk["word_count"].median())
                thr = blk["coverage"].quantile(0.10)
                blk["bottom"] = (blk["coverage"] <= thr).astype(float)
                blk = blk.join(groups, on="respondent_id")
                for aid, kind, col in analyses:
                    lab = blk[col].replace("", np.nan)
                    d = blk.assign(_lab=lab)
                    cnt = d["_lab"].value_counts()
                    for l, c in cnt.items():
                        sizes.append(dict(source=source, topic=topic, embedding_model=emb,
                                          summarizer=summ, analysis_id=aid, label=l, n=int(c)))
                    cands = ["individual", "organisation"] if kind == "secondary_respondent_type" else ["support", "oppose"]
                    if any(cnt.get(c, 0) < MIN_N for c in cands):
                        rows.append(dict(source=source, topic=topic, embedding_model=emb, summarizer=summ,
                                         analysis_id=aid, family=("primary_stance" if kind == "primary" else
                                         "secondary_stance" if kind == "secondary" else kind),
                                         note=f"skipped: a group has n<{MIN_N}", n_neutral=int(cnt.get('neutral', 0))))
                        continue
                    mn, mj = sorted(cands, key=lambda c: cnt[c])      # smaller first
                    seed = (SEED + sum(map(ord, f"{source}|{topic}|{emb}|{summ}|{aid}"))) % (2**32)
                    r = compare(d, "_lab", mn, mj, a.perms, a.boot, seed)
                    r.update(source=source, topic=topic, embedding_model=emb, summarizer=summ,
                             analysis_id=aid, n_neutral=int(cnt.get("neutral", 0)),
                             family={"primary": "primary_stance", "secondary": "secondary_stance"}.get(kind, kind),
                             bottom10_threshold=float(thr), note="")
                    rows.append(r)
                    print(f"{source} {topic} {emb} {summ} {aid}: diff={r['diff']:+.4f} p={r['p_perm']:.4f}", flush=True)
    if not n_inputs:
        sys.exit("No participant_coverage.csv inputs found yet; nothing to analyse.")
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan; res["adj_p_holm"] = np.nan
    for _, idx in res.groupby(["source", "summarizer", "family"]).groups.items():
        res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_perm"])
        res.loc[idx, "adj_p_holm"] = holm(res.loc[idx, "adj_p"])
    lead = ["source", "summarizer", "family", "topic", "embedding_model", "analysis_id"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(out / "results_all.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "group_sizes.csv", index=False)
    prim = res[(res.source == "official") & (res.family == "primary_stance")]
    prim.to_csv(out / "results_primary_official.csv", index=False)
    write_md(res, out / "RESULTS.md", a)


# ------------------------------------------------------------------- reporting
def f(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df, cols, heads):
    s = "| " + " | ".join(heads) + " |\n|" + "|".join(["---"] * len(heads)) + "|\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def fmt(df):
    d = df.copy()
    d["groups"] = d["minority_label"] + " (n=" + d["n_min"].astype("Int64").astype(str) + ") vs " + \
                  d["majority_label"] + " (n=" + d["n_maj"].astype("Int64").astype(str) + ")"
    d["diff_ci"] = d.apply(lambda r: f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", axis=1)
    d["adj"] = d.apply(lambda r: f"{r['adj_coef']:+.3f} [{r['adj_lo']:+.3f}, {r['adj_hi']:+.3f}]", axis=1)
    d["p"] = d["p_perm"].map(lambda x: f(x, 4)); d["ph"] = d["p_holm"].map(lambda x: f(x, 4))
    d["ap"] = d["adj_p"].map(lambda x: f(x, 4)); d["aph"] = d["adj_p_holm"].map(lambda x: f(x, 4))
    d["b10"] = d.apply(lambda r: f"{r['share_bottom10_min']:.1%} / {r['share_bottom10_maj']:.1%}", axis=1)
    d["low"] = d.apply(lambda r: (f"{r['lowcov_rate_min']:.1%} / {r['lowcov_rate_maj']:.1%}"
                                  if "lowcov_rate_min" in r and pd.notna(r.get("lowcov_rate_min")) else "n/a"), axis=1)
    return d


COLS = ["topic", "embedding_model", "analysis_id", "groups", "diff_ci", "p", "ph", "adj", "ap", "aph", "b10", "low"]
HEADS = ["Topic", "Embedding", "Analysis", "Groups (minority vs majority)", "Mean diff [95% CI]",
         "Perm p", "Holm p", "Length-adj. diff [95% CI]", "Adj. p", "Adj. Holm p",
         "Bottom-decile share (min / maj)", "Low-cov rate (min / maj)"]


def write_md(res, path, a):
    ok = res[res["p_perm"].notna()]
    L = ["# Do summaries under-cover minority stances? Coverage by respondent-defined group\n",
         "Groups come from participants' OWN closed answers (see MAPPING.json, frozen before any coverage was analysed) "
         "or from the individual/organisation field. No human labelling and no embedding clusters were used.\n",
         "## How to read this (guardrails)\n",
         "- Coverage is the maximum cosine similarity between a participant's response and any summary sentence. "
         "A group difference is a **semantic-coverage difference**, not proof that a view was misrepresented, ignored or suppressed.",
         "- Coverage rises with response length, so the length-adjusted estimate (OLS on group + log word count, HC3 SE) is the more conservative reading; "
         "if the raw and adjusted differences disagree, the raw gap is largely a length effect.",
         "- Difference = mean coverage of the MINORITY stance group minus the MAJORITY stance group (smaller / larger of support and oppose in this public subset, 1,809 of 2,207 responses). Negative = minority covered less. For respondent type, minority = organisation.",
         "- Neutral respondents are excluded from the stance tests. Stance is defined only among people who wrote a free-text answer to that question.",
         f"- Permutation p-values: {a.perms} respondent-level label permutations, two-sided, fixed seed. CIs: {a.boot} bootstrap resamples within group. "
         "Holm correction is applied within each family (source x summarizer x family); the primary family has one test per topic x embedding model.",
         "- Bottom decile = coverage at or below the 10th percentile across ALL respondents of that topic/embedding/summarizer.",
         "- Absence of a significant difference is not evidence of equal treatment; small minority groups have wide intervals.\n"]
    sections = [("Primary: official summary, minority vs majority stance", "official", "primary_stance"),
                ("Secondary: official summary, other linked closed questions", "official", "secondary_stance"),
                ("Secondary: official summary, organisation vs individual", "official", "secondary_respondent_type")]
    for title, src, fam in sections:
        d = ok[(ok.source == src) & (ok.family == fam)]
        L.append(f"## {title}\n")
        if d.empty:
            L.append("_No results (input missing or a group had too few respondents)._\n"); continue
        L.append(md_table(fmt(d), COLS, HEADS))
        sig = d[d["p_holm"] < 0.05]
        L.append(f"\nHolm-significant (<0.05) tests: {len(sig)} of {len(d)}. "
                 f"Tests where the minority group has lower raw mean coverage: {(d['diff'] < 0).sum()} of {len(d)}.\n")
    g = ok[ok.source == "generated"]
    L.append("## Generated (LLM) summaries (exploratory)\n")
    if g.empty:
        L.append("_No generated-summary coverage found._\n")
    else:
        L.append("Coverage is averaged over seeds per respondent before testing. Holm within summarizer x family.\n")
        for (summ, fam), d in g.groupby(["summarizer", "family"]):
            L.append(f"### {summ} - {fam}\n")
            L.append(md_table(fmt(d), COLS, HEADS))
    sk = res[res["note"].fillna("") != ""]
    if len(sk):
        L.append("## Skipped analyses\n")
        L.append(md_table(sk, ["source", "topic", "embedding_model", "analysis_id", "note"],
                          ["Source", "Topic", "Embedding", "Analysis", "Note"]))
    L.append("\nFull numeric output: results_all.csv, results_primary_official.csv, group_sizes.csv.\n")
    path.write_text("\n".join(L), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
