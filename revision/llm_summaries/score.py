#!/usr/bin/env python3
"""Embed generated summary sentences (same models/settings as the original pipeline) and
compute per-participant coverage = max cosine to any summary sentence (pipeline/04_transport.compute_coverage).

  revision/.venv/bin/python revision/llm_summaries/score.py [--topics ...]
Internal mode (spawned automatically, one subprocess per topic/embedding so each case's
analysis_io/config import is isolated):  score.py --dump-original TOPIC EMB
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

EMBED_CACHE = SCORE_DIR / "sentence_embeddings.sqlite"


def compute_coverage(inp, summ):
    """Verbatim logic of pipeline/04_transport.compute_coverage."""
    src = inp / (np.linalg.norm(inp, axis=1, keepdims=True) + 1e-10)
    tgt = summ / (np.linalg.norm(summ, axis=1, keepdims=True) + 1e-10)
    return (src @ tgt.T).max(axis=1)


# ------------------------------------------------------------ original blocks
def dump_original(topic, emb):
    import importlib
    case = CASES[topic]
    root, moddir = ANALYSIS_ROOT[case], PIPELINE_MODULE_DIR[case]
    os.environ["PROVENANCE_OUTPUT_DIR"] = str(root)
    sys.path.insert(0, str(moddir))
    os.chdir(moddir)
    analysis_io = importlib.import_module("analysis_io")
    blocks = analysis_io.embedding_blocks(topic, emb, root)
    clean = blocks["clean"]
    idc = pick_col(clean.columns, ID_CANDIDATES, what="id column of clean.parquet")
    out = SCORE_DIR / topic
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"_original_{emb}.npz", ids=np.array(clean[idc].astype(str).tolist()),
                        records=blocks["records"], summary=blocks["summary"])
    (out / f"_original_{emb}_sentences.json").write_text(json.dumps(blocks["summary_sentences"], indent=1))
    print(f"dumped original {topic}/{emb}: {len(clean)} participants, {len(blocks['summary'])} summary sentences")


def load_original(topic, emb):
    f = SCORE_DIR / topic / f"_original_{emb}.npz"
    if not f.exists():
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--dump-original", topic, emb], check=True)
    z = np.load(f)
    return z["ids"], z["records"], z["summary"]


# ----------------------------------------------------------- sentence embedding
_models = {}


def embed_openai(texts):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(max_retries=8)
    out = []
    for lo in range(0, len(texts), 512):
        batch = [t[:8000] for t in texts[lo:lo + 512]]          # same truncation as 02_prepare_benchmarks.embed_openai
        r = client.embeddings.create(model="text-embedding-3-large", input=batch)
        out.extend(d.embedding for d in r.data)
    return np.asarray(out, dtype=np.float32)


def embed_mpnet(texts):
    from sentence_transformers import SentenceTransformer
    if "mpnet" not in _models:
        _models["mpnet"] = SentenceTransformer("all-mpnet-base-v2")
    return np.asarray(_models["mpnet"].encode(texts, batch_size=64, show_progress_bar=False,
                                              convert_to_numpy=True), dtype=np.float32)


def embed_cached(emb, texts):
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(EMBED_CACHE, timeout=60)
    db.execute("CREATE TABLE IF NOT EXISTS e (emb TEXT, h TEXT, v BLOB, PRIMARY KEY (emb,h))")
    hs = [sha256(t) for t in texts]
    have = {}
    for h in set(hs):
        r = db.execute("SELECT v FROM e WHERE emb=? AND h=?", (emb, h)).fetchone()
        if r:
            have[h] = np.frombuffer(r[0], dtype=np.float32)
    todo = sorted({(h, t) for h, t in zip(hs, texts) if h not in have})
    if todo:
        print(f"   embedding {len(todo)} new sentences with {emb}", flush=True)
        vecs = (embed_openai if emb == "openai" else embed_mpnet)([t for _, t in todo])
        for (h, _), v in zip(todo, vecs):
            db.execute("INSERT OR REPLACE INTO e VALUES (?,?,?)", (emb, h, v.astype(np.float32).tobytes()))
            have[h] = v
        db.commit()
    db.close()
    return np.stack([have[h] for h in hs])


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=ALL_TOPICS, choices=ALL_TOPICS)
    ap.add_argument("--emb", nargs="+", default=EMB_MODELS, choices=EMB_MODELS)
    ap.add_argument("--dump-original", nargs=2, metavar=("TOPIC", "EMB"))
    args = ap.parse_args()
    if args.dump_original:
        return dump_original(*args.dump_original)
    import pandas as pd
    for topic in args.topics:
        rows = []
        try:
            for emb in args.emb:
                ids, records, off_summ = load_original(topic, emb)
                cov = compute_coverage(records, off_summ)
                rows.append(pd.DataFrame(dict(respondent_id=ids, embedding_model=emb, summarizer="official",
                                              seed="", coverage=cov)))
                print(f"[{topic}/{emb}] official mean coverage {cov.mean():.4f}")
                for slug in SLUGS:
                    for sf in sorted((OUT_DIR / topic / slug).glob("seed_*/summary.json")):
                        s = json.loads(sf.read_text())
                        sv = embed_cached(emb, s["sentences"])
                        c = compute_coverage(records, sv)
                        rows.append(pd.DataFrame(dict(respondent_id=ids, embedding_model=emb, summarizer=slug,
                                                      seed=s["seed"], coverage=c)))
                        print(f"[{topic}/{emb}] {slug} seed {s['seed']}: mean {c.mean():.4f} "
                              f"({s['n_sentences']} sent, {s['n_words']} words)")
        except Exception as e:
            print(f"SKIP {topic}: {type(e).__name__}: {e}", flush=True)
            continue
        out = SCORE_DIR / topic
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(rows).to_csv(out / "participant_coverage.csv", index=False)
        print(f"wrote {out/'participant_coverage.csv'}")


if __name__ == "__main__":
    main()
