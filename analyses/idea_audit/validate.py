"""Step 4: planted-ground-truth validation of the detector (no human annotation)."""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd

import common as C
from judge import Judge, prefetch_llm
from stats import COVER_LABELS, trigram_overlap, wilson

PARA_SYS = """Rewrite an idea as ONE sentence that could appear in an official consultation summary (e.g. starting 'Respondents ...' or 'Many respondents ...').
It must express exactly the same position and stance, but must NOT reuse wording from the original: avoid its distinctive words and never repeat any three consecutive words. {extra}
Return JSON: {{"sentence":"..."}}"""
FLIP_SYS = """Given an idea (with stance), write ONE sentence that could appear in an official consultation summary (e.g. starting 'Respondents ...') stating the directly OPPOSITE position on the same subject: support <-> oppose; a concern becomes the claim that the issue is not a concern / is overstated; a suggestion becomes rejection of that suggestion. Keep the same subject terms so it is a minimal reversal.
Return JSON: {"sentence":"..."}"""
EXTRA = ["", "Use clearly different vocabulary and sentence structure.", "Be abstract: describe the underlying principle in totally different words."]


def paraphrase(statement, stance):
    best = None
    for i, ex in enumerate(EXTRA):
        s = str(C.chat_json(PARA_SYS.format(extra=ex), f"Stance: {stance}\nIdea: {statement}", max_tokens=150).get("sentence", "")).strip()
        ov = trigram_overlap(s, statement)
        if s and (best is None or ov < best[1]):
            best = (s, ov)
        if s and ov == 0:
            break
    return best or ("", 99)


def flip(statement, stance):
    return str(C.chat_json(FLIP_SYS, f"Stance: {stance}\nIdea: {statement}", max_tokens=150).get("sentence", "")).strip()


def agg(k, n):
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n if n else float("nan")), "ci_lo": lo, "ci_hi": hi}


def cos_new(J_, statement, sentence):
    E = C.embed_texts([statement, sentence])
    return float(E[0] @ E[1])


def run(topic, thr, summaries, max_ideas, min_resp, workers):
    C.set_step("validate")
    d = C.topic_dir(topic)
    ideas = pd.read_parquet(d / f"ideas_{C.tag_for(thr)}.parquet")
    ideas = ideas[ideas.n_resp >= min_resp].set_index("idea_id")
    jd = pd.read_parquet(d / f"judgments_{C.tag_for(thr)}.parquet")
    sums = C.load_summaries(topic, summaries)
    J = Judge()
    rows = []
    rng = np.random.RandomState(0)
    for name, sents in sums.items():
        base = jd[jd.summary == name].set_index("idea_id")
        base = base[base.index.isin(ideas.index)]

        def sample(ids):
            ids = sorted(ids)
            if len(ids) > max_ideas:
                ids = sorted(rng.choice(ids, max_ideas, replace=False).tolist())
            return ids

        # (i) planted omission: remove every sentence that individually entails the idea (>=0.5), else the top one
        cov = sample(base.index[base.label == "COVERED"])
        jobs, plan = [], {}
        for iid in cov:
            idea = ideas.loc[iid]
            ev = J.evaluate(idea.canonical, idea.stance, sents)
            se = np.array(ev["nli_sent_entail"])
            drop = set(np.where(se >= 0.5)[0].tolist()) or {int(se.argmax())}
            kept = [s for i, s in enumerate(sents) if i not in drop]
            if kept:
                plan[iid] = (kept, len(drop))
                jobs.append((idea.canonical, idea.stance, kept))
        prefetch_llm(jobs, workers)
        for iid, (kept, nd) in plan.items():
            idea = ideas.loc[iid]
            r = J.evaluate(idea.canonical, idea.stance, kept)
            rows.append({"test": "omission", "summary": name, "idea_id": iid, "n_removed": nd, **{k: r[k] for k in ("nli_entail", "nli_contra", "llm", "label", "cos_max")}})
        # (ii)+(iii) planted inclusion / stance flip on ideas MISSED in the base summary
        mis = sample(base.index[base.label == "MISSED"])
        gen = C.pmap(lambda iid: (paraphrase(ideas.loc[iid].canonical, ideas.loc[iid].stance),
                                  flip(ideas.loc[iid].canonical, ideas.loc[iid].stance)), mis, workers, "plant-gen")
        jobs = []
        for iid, ((p, _), f) in zip(mis, gen):
            if p:
                jobs.append((ideas.loc[iid].canonical, ideas.loc[iid].stance, sents + [p]))
            if f:
                jobs.append((ideas.loc[iid].canonical, ideas.loc[iid].stance, sents + [f]))
        prefetch_llm(jobs, workers)
        for iid, ((p, ov), f) in zip(mis, gen):
            idea = ideas.loc[iid]
            if p:
                r = J.evaluate(idea.canonical, idea.stance, sents + [p])
                rows.append({"test": "inclusion", "summary": name, "idea_id": iid, "planted": p, "trigram_overlap": ov,
                             "cos_new": cos_new(J, idea.canonical, p), **{k: r[k] for k in ("nli_entail", "nli_contra", "llm", "label", "cos_max")}})
            if f:
                r = J.evaluate(idea.canonical, idea.stance, sents + [f])
                rows.append({"test": "flip", "summary": name, "idea_id": iid, "planted": f,
                             "cos_new": cos_new(J, idea.canonical, f), **{k: r[k] for k in ("nli_entail", "nli_contra", "llm", "label", "cos_max")}})
        print(f"  [{topic}/{name}] omission n={len(plan)}, inclusion/flip n={len(mis)}")

    # negative control: unrelated topic's official summary
    oname, osents = C.other_country_summary(topic)
    ids = sorted(ideas.index)
    if len(ids) > max_ideas:
        ids = sorted(rng.choice(ids, max_ideas, replace=False).tolist())
    prefetch_llm([(ideas.loc[i].canonical, ideas.loc[i].stance, osents) for i in ids], workers)
    for iid in ids:
        idea = ideas.loc[iid]
        r = J.evaluate(idea.canonical, idea.stance, osents)
        rows.append({"test": "negctrl", "summary": f"unrelated:{oname}", "idea_id": iid, **{k: r[k] for k in ("nli_entail", "nli_contra", "llm", "label", "cos_max")}})
    df = pd.DataFrame(rows)
    df["topic"] = topic
    df.to_parquet(d / f"validation_rows_{C.tag_for(thr)}.parquet", index=False)
    res = summarise(df)
    (d / f"validation_{C.tag_for(thr)}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1)[:3000])


def summarise(df):
    res = {}
    cov = lambda x: x.llm.isin(COVER_LABELS)
    o = df[df.test == "omission"]
    if len(o):
        res["omission (planted deletion): detected as absent"] = {
            "combined rule -> MISSED": agg((o.label == "MISSED").sum(), len(o)),
            "NLI only (entail<0.5)": agg((o.nli_entail < 0.5).sum(), len(o)),
            "LLM only (not_expressed)": agg((o.llm == "not_expressed").sum(), len(o)),
            "combined: not COVERED": agg((o.label != "COVERED").sum(), len(o))}
    i = df[df.test == "inclusion"]
    tau = None
    if len(i):
        tau = float(np.percentile(i.cos_new, 10))      # cosine threshold giving 90% recall on planted inclusions (favourable to cosine)
        res["inclusion (planted paraphrase): detected as covered"] = {
            "combined rule -> COVERED": agg((i.label == "COVERED").sum(), len(i)),
            "NLI only (entail>=0.5)": agg((i.nli_entail >= 0.5).sum(), len(i)),
            "LLM only (expressed/partially)": agg(cov(i).sum(), len(i)),
            f"cosine only (cos>=tau={tau:.3f}, tau = 10th pct of planted paraphrases)": agg((i.cos_new >= tau).sum(), len(i)),
            "median trigram overlap with statement": float(i.trigram_overlap.median())}
    f = df[df.test == "flip"]
    if len(f):
        r = {"combined rule -> COVERED (false coverage)": agg((f.label == "COVERED").sum(), len(f)),
             "NLI only (entail>=0.5)": agg((f.nli_entail >= 0.5).sum(), len(f)),
             "LLM only (expressed/partially)": agg(cov(f).sum(), len(f)),
             "combined -> STANCE_MISMATCH": agg((f.label == "STANCE_MISMATCH").sum(), len(f))}
        if tau is not None:
            r[f"cosine only (cos>=tau={tau:.3f}) false coverage"] = agg((f.cos_new >= tau).sum(), len(f))
        res["stance flip (planted opposite): false coverage"] = r
    n = df[df.test == "negctrl"]
    if len(n):
        res["negative control (unrelated summary): called MISSED"] = {
            "combined rule -> MISSED": agg((n.label == "MISSED").sum(), len(n)),
            "NLI only (entail<0.5)": agg((n.nli_entail < 0.5).sum(), len(n)),
            "LLM only (not_expressed)": agg((n.llm == "not_expressed").sum(), len(n))}
    return res


def main():
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--summaries", choices=["official", "all"], default="official")
    ap.add_argument("--max-ideas", type=int, default=150, help="max ideas per test per summary (seeded sample)")
    ap.add_argument("--min-resp", type=int, default=10)
    a = ap.parse_args()
    for t in a.topics:
        run(t, a.threshold, a.summaries, a.max_ideas, a.min_resp, a.workers)
    C.print_costs()


if __name__ == "__main__":
    main()
