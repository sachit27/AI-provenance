"""Step 2: embed claims, cluster WITHIN stance (agglomerative, cosine, average linkage), keep recurring ideas."""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

import common as C

MIN_KEEP = 5                      # stored; the a-priori analysis threshold is >=10 (sens. >=5, >=20)
SENS_THRESHOLDS = (0.25, 0.30, 0.35)
SENS_MIN = (5, 10, 20)
MAX_GROUP = 25000

SYSTEM = """You write a canonical statement for a cluster of similar claims from consultation responses.
Write ONE self-contained, neutral sentence capturing the position they share (no 'respondents', no 'some people'). Stay as specific as the claims are; do not generalise beyond them and do not add content. Preserve the stance exactly: support = favours something, oppose = argues against it, concern = raises a risk/harm, suggestion = proposes an action.
Return JSON: {"statement":"..."}"""


def cluster(X, thr):
    if len(X) == 1:
        return np.zeros(1, dtype=int)
    if len(X) > MAX_GROUP:
        raise SystemExit(f"stance group of {len(X)} claims too large for O(n^2) clustering")
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=thr).fit_predict(X)


def run(topic, thr, include_other, workers, sens):
    C.set_step("ideas")
    claims = pd.read_parquet(C.topic_dir(topic) / "claims.parquet")
    n_resp_total = len(C.load_responses(topic))
    if not include_other:
        claims = claims[claims.stance != "other"]
    claims = claims.reset_index(drop=True)
    X = C.embed_texts(claims.claim.tolist())
    groups = {s: np.where(claims.stance.values == s)[0] for s in claims.stance.unique()}

    if sens:
        table = {}
        for t in SENS_THRESHOLDS:
            cnt = {m: 0 for m in SENS_MIN}
            for s, idx in groups.items():
                lab = cluster(X[idx], t)
                nr = pd.Series(claims.rid.values[idx]).groupby(lab).nunique()
                for m in SENS_MIN:
                    cnt[m] += int((nr >= m).sum())
            table[f"{t:.2f}"] = {f">={m}": cnt[m] for m in SENS_MIN}
        (C.topic_dir(topic) / "sensitivity_idea_counts.json").write_text(json.dumps(table, indent=2))
        print(f"[{topic}] idea counts by threshold x min-respondents: {table}")

    recs = []
    for s, idx in groups.items():
        lab = cluster(X[idx], thr)
        for c in np.unique(lab):
            mem = idx[lab == c]
            rids = claims.rid.values[mem]
            nr = len(set(rids))
            if nr < MIN_KEEP:
                continue
            cen = X[mem].mean(0)
            cen /= np.linalg.norm(cen)
            order = mem[np.argsort(-(X[mem] @ cen))]
            recs.append((s, nr, order, mem))
    recs.sort(key=lambda r: (-r[1], r[0], r[2][0]))
    print(f"[{topic}] {len(recs)} ideas with >={MIN_KEEP} respondents at threshold {thr}")

    def prep(r):
        s, nr, order, mem = r
        seen, chosen = set(), []
        for i in order:                    # one claim per respondent, most central first
            if claims.rid.values[i] not in seen:
                seen.add(claims.rid.values[i])
                chosen.append(i)
            if len(chosen) == 12:
                break
        user = f"Stance: {s}\nClaims:\n" + "\n".join(f"- {claims.claim.values[i]}" for i in chosen)
        st = str(C.chat_json(SYSTEM, user, max_tokens=300).get("statement", "")).strip()
        if not st:
            st = claims.claim.values[chosen[0]]
        quotes, seen = [], set()
        for ver in (True, False):
            for i in order:
                rid = claims.rid.values[i]
                if rid in seen or len(quotes) == 3:
                    continue
                if ver and not claims.quote_verified.values[i]:
                    continue
                q = claims.quote.values[i] or claims.claim.values[i]
                quotes.append({"rid": rid, "quote": q, "verified": bool(claims.quote_verified.values[i])})
                seen.add(rid)
        return st, quotes

    prepped = C.pmap(prep, recs, workers, "canonical")
    rows = []
    for k, ((s, nr, order, mem), (st, quotes)) in enumerate(zip(recs, prepped)):
        rows.append({"idea_id": f"{topic}-{C.tag_for(thr)}-{k:04d}", "topic": topic, "threshold": thr, "stance": s,
                     "n_claims": len(mem), "n_resp": nr, "pct_resp": nr / n_resp_total, "canonical": st,
                     "quotes": json.dumps(quotes, ensure_ascii=False),
                     "member_rids": json.dumps(sorted(set(claims.rid.values[mem].tolist()))),
                     "member_claims": json.dumps([claims.claim.values[i] for i in order[:12]], ensure_ascii=False)})
    out = pd.DataFrame(rows)
    p = C.topic_dir(topic) / f"ideas_{C.tag_for(thr)}.parquet"
    out.to_parquet(p, index=False)
    print(f"[{topic}] wrote {p}; >=10 respondents: {(out.n_resp >= 10).sum()}")


def main():
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--include-other", action="store_true", help="also cluster stance=='other' (default: excluded)")
    ap.add_argument("--sensitivity-counts", action="store_true", help="also tabulate idea counts at 0.25/0.30/0.35 x min 5/10/20")
    a = ap.parse_args()
    for t in a.topics:
        run(t, a.threshold, a.include_other, a.workers, a.sensitivity_counts)
    C.print_costs()


if __name__ == "__main__":
    main()
