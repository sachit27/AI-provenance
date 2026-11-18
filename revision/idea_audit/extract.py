"""Step 1: atomic claim extraction (gpt-4o-mini, temp 0, JSON mode, sqlite-cached, resumable)."""
from __future__ import annotations
import argparse
import json

import pandas as pd

import common as C
from stats import quote_verified

SYSTEM = """You extract atomic claims from public-consultation responses.
For each response return the distinct ideas/positions the respondent raises.
Rules:
- Each claim is ONE self-contained, neutral sentence (no pronouns referring to the text, no 'the respondent'), e.g. "AI literacy should be taught in primary schools."
- Do not merge different ideas; do not invent anything not in the response. Skip greetings, filler and off-topic remarks. A response may have 0 claims. At most 8 claims per response.
- stance is one of: support (favours/endorses something), oppose (rejects/argues against something), concern (raises a risk or harm), suggestion (proposes an action or policy), other.
- target: a short phrase (2-6 words) naming what the claim is about.
- quote: a SHORT (<=200 chars) contiguous span copied EXACTLY, character for character, from the response that supports the claim.
Return JSON: {"results":[{"id":"<id as given>","claims":[{"claim":"...","stance":"...","target":"...","quote":"..."}]}]} with one entry per input response, in order."""

MAX_CHARS_PER_BATCH = 5000
MAX_ITEMS = 8
TRUNC = 6000


def make_batches(df):
    batches, cur, chars = [], [], 0
    for r in df.itertuples():
        n = min(len(r.text), TRUNC)
        if cur and (chars + n > MAX_CHARS_PER_BATCH or len(cur) >= MAX_ITEMS):
            batches.append(cur)
            cur, chars = [], 0
        cur.append((r.rid, r.text[:TRUNC]))
        chars += n
    if cur:
        batches.append(cur)
    return batches


def call(question, batch):
    payload = {"question": question,
               "responses": [{"id": f"r{i+1}", "text": t} for i, (_, t) in enumerate(batch)]}
    out = C.chat_json(SYSTEM, "Consultation question(s):\n" + question + "\n\nResponses (JSON):\n"
                      + json.dumps(payload, ensure_ascii=False), max_tokens=4000)
    got = {}
    results = out.get("results", []) if isinstance(out, dict) else []
    if not isinstance(results, list):
        results = []
    for item in results:
        if not isinstance(item, dict):
            continue  # malformed item; the missing id is retried alone below
        claims = item.get("claims") or []
        if not isinstance(claims, list):
            claims = []
        got[str(item.get("id"))] = [c for c in claims if isinstance(c, dict) and c.get("claim")]
    return got


def process_batch(args):
    question, batch = args
    got = call(question, batch)
    res = {}
    for i, (rid, text) in enumerate(batch):
        k = f"r{i+1}"
        if k in got:
            res[rid] = got[k]
        else:   # model dropped an id -> retry that response alone
            res[rid] = call(question, [(rid, text)]).get("r1", [])
    return res


def run(topic, limit, workers):
    C.set_step("extract")
    df = C.load_responses(topic)
    if limit:
        df = df.head(limit)
    texts = dict(zip(df.rid, df.text))
    question = C.question_text(topic)
    batches = make_batches(df)
    print(f"[{topic}] {len(df)} responses in {len(batches)} calls")
    results = C.pmap(process_batch, [(question, b) for b in batches], workers, "extract")
    rows = []
    for res in results:
        for rid, claims in res.items():
            k = 0
            for c in claims:
                if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
                    continue
                stance = str(c.get("stance", "other")).lower().strip()
                stance = stance if stance in C.STANCES else "other"
                q = str(c.get("quote", "") or "")
                rows.append({"claim_id": f"{rid}#{k}", "rid": rid, "claim": str(c["claim"]).strip(),
                             "stance": stance, "target": str(c.get("target", "")).strip(),
                             "quote": q, "quote_verified": bool(quote_verified(q, texts[rid]))})
                k += 1
    out = pd.DataFrame(rows)
    p = C.topic_dir(topic) / "claims.parquet"
    out.to_parquet(p, index=False)
    print(f"[{topic}] {len(out)} claims from {out.rid.nunique()}/{len(df)} respondents -> {p}")
    print(f"  quote verified as substring: {out.quote_verified.mean():.1%}; stances: {out.stance.value_counts().to_dict()}")
    return out


def main():
    ap = C.add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--limit", type=int, default=0, help="pilot: only first N responses (cached; rerun without it later)")
    a = ap.parse_args()
    for t in a.topics:
        run(t, a.limit, a.workers)
    C.print_costs()


if __name__ == "__main__":
    main()
