#!/usr/bin/env python3
"""Sensitivity: restore answers removed by the stage-01 relevance screen.

Run from the canadian_data root:
    analyses/.venv/bin/python analyses/scotland_analysis/sensitivity_restored/restore_sensitivity.py

Restored rows = filter_audit.parquet rows with exclusion_reason in
{relevance_auto_reject, relevance_llm_reject} (NOT empty / french / short / spam).
Text is rebuilt from inputs/<q>.csv at audit['source_row'] exactly as in
01_preprocess.py (non-null cells joined with a space, stripped).
Embeddings use the pipeline's own get_embeddings / embed_openai / embed_mpnet /
normalize (imported from 02_prepare_benchmarks.py). Writes only into this folder.
"""
import sys
sys.dont_write_bytecode = True  # do not create __pycache__ in the pipeline folder
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SC = HERE.parent
ROOT = SC.parents[1]
ANALYSIS = SC / "analysis"
PIPE = SC / "pipeline"
os.environ.setdefault("PROVENANCE_OUTPUT_DIR", str(ANALYSIS))
sys.path.insert(0, str(PIPE))

import analysis_io  # noqa: E402  (pipeline/analysis_io.py)
from config import MIN_WORD_COUNT  # noqa: E402

_spec = importlib.util.spec_from_file_location("prep02", PIPE / "02_prepare_benchmarks.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)  # provides get_embeddings, normalize, gini, TAIL_FRACTION

QUESTIONS = ["q4", "q7", "q20", "q21", "q22", "q23"]
MODELS = ["openai", "mpnet"]
REL_REASONS = {"relevance_auto_reject", "relevance_llm_reject"}
EXPECTED = {"q4": 0, "q7": 3, "q20": 27, "q21": 14, "q22": 32, "q23": 10}
PRIMARY = {"q4": "g_q4_level", "q7": "g_q7_Q5", "q23": "g_q23_Q19"}
SEED, B_PERM, MIN_N = 1729, 10000, 10


def restored_records(q):
    audit = pd.read_parquet(ANALYSIS / q / "filter_audit.parquet")
    rem = audit[audit["exclusion_reason"].isin(REL_REASONS)].copy()
    if rem.empty:
        return pd.DataFrame(columns=["source_row", "respondent_id", "text", "reason"])
    raw = pd.read_csv(SC / "inputs" / f"{q}.csv", low_memory=False)
    cols = [c for c in raw.columns if c != "Internal ID"]
    rows = []
    for _, r in rem.iterrows():
        row = raw.iloc[int(r["source_row"])]
        assert str(row["Internal ID"]) == str(r["Internal ID"]), (q, r["source_row"])
        # same concatenation as 01_preprocess.py
        text = " ".join(str(row[c]) for c in cols if pd.notna(row[c]) and str(row[c]).strip()).strip()
        rows.append(dict(source_row=int(r["source_row"]), respondent_id=str(r["Internal ID"]),
                         text=text, reason=r["exclusion_reason"]))
    out = pd.DataFrame(rows)
    assert (out["text"].str.split().str.len() >= MIN_WORD_COUNT).all(), f"{q}: short text restored"
    return out


def restored_embeddings(q, model, texts):
    path = HERE / f"restored_{q}_{model}.npz"
    emb = prep.get_embeddings(model, texts, path)  # embeds + caches (digest-checked)
    return prep.normalize(emb)


def metrics(cov, thr):
    cov = np.asarray(cov, float)
    k = max(1, int(np.ceil(prep.TAIL_FRACTION * len(cov))))
    return dict(N=len(cov), mean_cov=cov.mean(),
                bottom_decile_mean=float(np.partition(cov, k - 1)[:k].mean()),
                gini=prep.gini(cov), low_cov_rate=float(np.mean(cov < thr)))


def perm_p(y, n_min, B, rng, obs):  # same logic as stance_groups/analyze.py
    n = len(y); tot = y.sum(); n_maj = n - n_min
    cnt = 0; done = 0
    while done < B:
        c = min(500, B - done)
        idx = np.argpartition(rng.random((c, n)), n_min - 1, axis=1)[:, :n_min]
        s = y[idx].sum(axis=1)
        d = s / n_min - (tot - s) / n_maj
        cnt += int((np.abs(d) >= abs(obs) - 1e-12).sum()); done += c
    return (cnt + 1) / (B + 1)


def group_test(ids, cov, labels, min_lab, maj_lab):
    lab = pd.Series(labels, index=ids)
    a = np.asarray(cov)[(lab == min_lab).to_numpy()]
    b = np.asarray(cov)[(lab == maj_lab).to_numpy()]
    y = np.r_[a, b]
    obs = a.mean() - b.mean()
    p = perm_p(y, len(a), B_PERM, np.random.default_rng(SEED), obs)
    return dict(n_min=len(a), n_maj=len(b), mean_min=a.mean(), mean_maj=b.mean(), diff=obs, p_perm=p)


def main():
    groups = pd.read_csv(ROOT / "analyses/stance_groups/respondent_groups.csv", dtype=str).set_index("respondent_id")
    res, stance, restored_log = [], [], []
    for q in QUESTIONS:
        rr = restored_records(q)
        print(f"{q}: restored {len(rr)} (expected {EXPECTED[q]})", flush=True)
        restored_log.append(dict(question=q, n_restored=len(rr), expected=EXPECTED[q]))
        for model in MODELS:
            blk = analysis_io.embedding_blocks(q, model, ANALYSIS)
            rec, summ, clean = blk["records"], blk["summary"], blk["clean"]
            ids_ret = clean["Internal ID"].astype(str).to_numpy()
            cov_ret = (rec @ summ.T).max(axis=1)
            bj = json.loads((ANALYSIS / q / f"benchmark_{model}.json").read_text())
            thr = float(bj["frozen_exclusion_threshold"])
            m_ret = metrics(cov_ret, thr)
            if len(rr):
                emb = restored_embeddings(q, model, rr["text"].tolist())
                cov_res = (emb @ summ.T).max(axis=1)
                cov_all = np.r_[cov_ret, cov_res]
                ids_all = np.r_[ids_ret, rr["respondent_id"].to_numpy()]
                m_all = metrics(cov_all, thr)
                m_res = metrics(cov_res, thr)
            else:
                cov_res = np.array([]); cov_all, ids_all = cov_ret, ids_ret
                m_all = m_ret; m_res = dict(N=0, mean_cov=np.nan, low_cov_rate=np.nan)
            res.append(dict(question=q, model=model, threshold=thr,
                            N_retained=m_ret["N"], N_plus_restored=m_all["N"], N_restored=m_res["N"],
                            mean_retained=m_ret["mean_cov"], mean_plus=m_all["mean_cov"],
                            mean_restored_only=m_res["mean_cov"],
                            bottom10_retained=m_ret["bottom_decile_mean"], bottom10_plus=m_all["bottom_decile_mean"],
                            gini_retained=m_ret["gini"], gini_plus=m_all["gini"],
                            lowcov_retained=m_ret["low_cov_rate"], lowcov_plus=m_all["low_cov_rate"],
                            lowcov_restored_only=m_res["low_cov_rate"]))
            if q in PRIMARY:
                col = PRIMARY[q]
                lab_ret = groups[col].reindex(ids_ret).to_numpy()
                lab_all = groups[col].reindex(ids_all).to_numpy()
                cnt = pd.Series(lab_ret).value_counts()
                if any(cnt.get(c, 0) < MIN_N for c in ("support", "oppose")):
                    continue
                mn, mj = sorted(["support", "oppose"], key=lambda c: cnt[c])  # minority fixed from retained set
                for name, ids, cov, lab in (("retained", ids_ret, cov_ret, lab_ret),
                                            ("retained+restored", ids_all, cov_all, lab_all)):
                    r = group_test(ids, cov, lab, mn, mj)
                    stance.append(dict(question=q, model=model, analysis=col, set=name,
                                       minority=mn, majority=mj, **r))
                    print(f"  {q} {model} {name}: diff={r['diff']:+.4f} p={r['p_perm']:.4f}", flush=True)
    R = pd.DataFrame(res); S = pd.DataFrame(stance)
    R.to_csv(HERE / "results.csv", index=False)
    S.to_csv(HERE / "results_stance.csv", index=False)
    write_md(R, S, pd.DataFrame(restored_log))


def write_md(R, S, L):
    def f(x, d=3):
        return "" if pd.isna(x) else f"{x:.{d}f}"
    out = ["# Restored-answer sensitivity (Scottish official-summary analysis)\n",
           "Restored = rows removed by the stage-01 relevance screen (relevance_auto_reject / relevance_llm_reject). "
           "Coverage = max cosine to the official summary sentences. Low-coverage uses the FIXED retained-analysis "
           "threshold (benchmark_<model>.json frozen_exclusion_threshold). Cross-fitted extractive benchmarks are not recomputed (see README).\n",
           "## Restored counts\n", "| Question | Restored | Expected |\n|---|---|---|"]
    out += [f"| {r.question} | {r.n_restored} | {r.expected} |" for r in L.itertuples()]
    out += ["\n## Coverage metrics: retained vs retained+restored\n",
            "| Q | Model | N ret -> +rest | Mean | Bottom-10% mean | Gini | Low-cov rate | Restored-only mean (low-cov) |",
            "|---|---|---|---|---|---|---|---|"]
    for r in R.itertuples():
        out.append(f"| {r.question} | {r.model} | {r.N_retained} -> {r.N_plus_restored} | "
                   f"{f(r.mean_retained)} -> {f(r.mean_plus)} | {f(r.bottom10_retained)} -> {f(r.bottom10_plus)} | "
                   f"{f(r.gini_retained, 4)} -> {f(r.gini_plus, 4)} | {f(r.lowcov_retained)} -> {f(r.lowcov_plus)} | "
                   f"{f(r.mean_restored_only)} ({f(r.lowcov_restored_only)}) |")
    out += ["\n## Closed-answer groups: minority - majority mean coverage (10,000 permutations, seed 1729)\n",
            "| Q | Model | Analysis | Set | Minority (n) vs majority (n) | Diff | Perm p |", "|---|---|---|---|---|---|---|"]
    for r in S.itertuples():
        out.append(f"| {r.question} | {r.model} | {r.analysis} | {r.set} | {r.minority} ({r.n_min}) vs {r.majority} ({r.n_maj}) | "
                   f"{r.diff:+.4f} | {r.p_perm:.4f} |")
    mean_shift = (R["mean_plus"] - R["mean_retained"]).abs().max()
    gini_shift = (R["gini_plus"] - R["gini_retained"]).abs().max()
    flips = 0
    if len(S):
        w = S.pivot_table(index=["question", "model"], columns="set", values=["diff", "p_perm"])
        flips = int((((w["p_perm"]["retained"] < .05) != (w["p_perm"]["retained+restored"] < .05)) |
                     (np.sign(w["diff"]["retained"]) != np.sign(w["diff"]["retained+restored"]))).sum())
    out += ["\n## Summary\n",
            f"Restoring the {int(L['n_restored'].sum())} screened answers changes the mean coverage by at most {mean_shift:.4f} and the Gini by at most {gini_shift:.4f} across questions and models. "
            f"The restored answers alone have the mean coverage shown in the table (compare with the retained mean), and the low-coverage rates move only by the amounts listed. "
            f"Of the {len(S)//2} minority-vs-majority comparisons, {flips} change in sign or in p<0.05 significance after restoration, so the conclusions "
            + ("are unchanged." if flips == 0 else "should be checked against the table above (some flips).")]
    (HERE / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote RESULTS.md, results.csv, results_stance.csv")


if __name__ == "__main__":
    main()
