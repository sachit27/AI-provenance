"""Shared I/O, sqlite cache, OpenAI wrappers, cost tallies for the idea-omission audit."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IA = Path(__file__).resolve().parent
REV = IA.parent
BASE = REV.parent                      # canadian_data
PIPE = BASE / "src"
WORK = IA / "work"
TOPICS = {"education": "canada", "trust": "canada", "q7": "scotland", "q21": "scotland"}
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
PRICE = {CHAT_MODEL: (0.15, 0.60), EMBED_MODEL: (0.13, 0.0)}   # USD per 1M tokens (in, out)
STANCES = ("support", "oppose", "concern", "suggestion", "other")

try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except Exception:   # pragma: no cover
    pass


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ───────────────────────── sqlite cache + usage ─────────────────────────
class Cache:
    def __init__(self, path: Path | None = None):
        path = path or (WORK / "cache.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("create table if not exists kv(k text primary key, v text)")
        self.db.execute("create table if not exists emb(k text primary key, v blob)")
        self.db.execute("create table if not exists usage(id integer primary key autoincrement, "
                        "step text, model text, tin integer, tout integer)")
        self.db.commit()

    def get(self, k):
        with self.lock:
            r = self.db.execute("select v from kv where k=?", (k,)).fetchone()
        return None if r is None else json.loads(r[0])

    def set(self, k, v):
        with self.lock:
            self.db.execute("insert or replace into kv values(?,?)", (k, json.dumps(v, ensure_ascii=False)))
            self.db.commit()

    def set_many(self, items):
        with self.lock:
            self.db.executemany("insert or replace into kv values(?,?)",
                                [(k, json.dumps(v, ensure_ascii=False)) for k, v in items])
            self.db.commit()

    def get_many(self, keys):
        out = {}
        keys = list(keys)
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = "select k,v from kv where k in (%s)" % ",".join("?" * len(chunk))
                for k, v in self.db.execute(q, chunk):
                    out[k] = json.loads(v)
        return out

    def get_emb(self, keys):
        import numpy as np
        out = {}
        keys = list(keys)
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = "select k,v from emb where k in (%s)" % ",".join("?" * len(chunk))
                for k, v in self.db.execute(q, chunk):
                    out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def set_emb(self, items):
        with self.lock:
            self.db.executemany("insert or replace into emb values(?,?)", [(k, v.astype("float32").tobytes()) for k, v in items])
            self.db.commit()

    def log(self, step, model, tin, tout):
        with self.lock:
            self.db.execute("insert into usage(step,model,tin,tout) values(?,?,?,?)", (step, model, int(tin), int(tout)))
            self.db.commit()

    def costs(self):
        with self.lock:
            rows = self.db.execute("select step,model,sum(tin),sum(tout),count(*) from usage group by step,model").fetchall()
        return rows


_CACHE = None


def cache() -> Cache:
    global _CACHE
    if _CACHE is None:
        _CACHE = Cache()
    return _CACHE


def print_costs(title="Token / cost tally (all runs, from cache DB; cached calls cost nothing)"):
    print(f"\n== {title} ==")
    tot = 0.0
    for step, model, tin, tout, n in cache().costs():
        pin, pout = PRICE.get(model, (0.15, 0.60))
        c = tin / 1e6 * pin + tout / 1e6 * pout
        tot += c
        print(f"  {step:<10} {model:<24} calls={n:<6} in={tin:<10} out={tout:<9} ${c:.4f}")
    print(f"  TOTAL ${tot:.4f}")


_CLIENT = None
_STEP = ["misc"]


def set_step(name: str):
    _STEP[0] = name


def client():
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set (put it in canadian_data/.env)")
        _CLIENT = OpenAI(max_retries=2, timeout=120)
    return _CLIENT


def chat_json(system: str, user: str, model: str = CHAT_MODEL, max_tokens: int = 4000, tag: str = "") -> dict:
    """gpt-4o-mini, temperature 0, JSON mode, sqlite-cached. `tag` only perturbs the cache key."""
    key = "chat|" + sha(model, system, user, tag)
    hit = cache().get(key)
    if hit is not None:
        return hit
    last = None
    for attempt in range(6):
        try:
            r = client().chat.completions.create(
                model=model, temperature=0, seed=0, max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            cache().log(_STEP[0], model, r.usage.prompt_tokens, r.usage.completion_tokens)
            out = json.loads(r.choices[0].message.content)
            cache().set(key, out)
            return out
        except json.JSONDecodeError as e:
            last = e
            cache().log(_STEP[0], model, 0, 0)
        except Exception as e:   # rate limits, transient
            last = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"chat_json failed: {last}")


def embed_texts(texts, batch: int = 128):
    """text-embedding-3-large, L2-normalised float32 matrix, cached per text."""
    import numpy as np
    texts = [t if t.strip() else "." for t in (x[:8000] for x in texts)]
    keys = ["emb|" + sha(EMBED_MODEL, t) for t in texts]
    have = cache().get_emb(set(keys))
    todo = {}
    for k, t in zip(keys, texts):
        if k not in have and k not in todo:
            todo[k] = t
    items = list(todo.items())
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        for attempt in range(6):
            try:
                r = client().embeddings.create(model=EMBED_MODEL, input=[t for _, t in chunk])
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 ** attempt))
        cache().log("embed", EMBED_MODEL, r.usage.total_tokens, 0)
        vecs = [np.asarray(d.embedding, dtype=np.float32) for d in r.data]
        vecs = [v / max(np.linalg.norm(v), 1e-12) for v in vecs]
        cache().set_emb(list(zip([k for k, _ in chunk], vecs)))
        for (k, _), v in zip(chunk, vecs):
            have[k] = v
        if len(items) > batch:
            print(f"  embedded {min(i + batch, len(items))}/{len(items)}", flush=True)
    return np.stack([have[k] for k in keys]) if keys else np.zeros((0, 3072), dtype=np.float32)


def pmap(fn, items, workers: int = 8, label: str = ""):
    items = list(items)
    out = [None] * len(items)
    done = [0]
    lock = threading.Lock()

    def run(i):
        out[i] = fn(items[i])
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(items):
                print(f"  {label} {done[0]}/{len(items)}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(len(items))))
    return out


# ───────────────────────── data loading ─────────────────────────
ID_CANDS = ["respondent_id", "Internal ID", "participant_id", "response_id", "id", "ID", "record_id", "row_id", "uuid"]
TXT_CANDS = ["text", "response", "clean_text", "response_text", "answer", "comment", "content", "cleaned_text", "body"]


def split_sentences(text: str):
    """Same splitter as src/analysis_io.split_summary."""
    return [p.strip() for p in re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text.strip()) if p.strip()]


def load_responses(topic: str):
    """Return DataFrame[rid, text] of retained responses (auto-detected columns)."""
    import pandas as pd
    if TOPICS[topic] == "canada":
        out_dir = Path(os.environ.get("PROVENANCE_OUTPUT_DIR", str(PIPE / "analysis")))
        p = out_dir / topic / "clean.parquet"
    else:
        p = REV / "scotland_analysis" / "analysis" / topic / "clean.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        src = str(p)
    elif TOPICS[topic] == "scotland":
        rec = REV / "scotland" / "datasets" / topic / "records.jsonl"
        if not rec.exists():
            sys.exit(f"Missing {p} and {rec}")
        print(f"WARNING: {p} missing; falling back to {rec} (all records, not the retained subset)")
        df = pd.DataFrame([json.loads(s) for s in rec.read_text().splitlines() if s.strip()])
        src = str(rec)
    else:
        sys.exit(f"Missing {p}; run the pipeline first")
    idc = next((c for c in ID_CANDS if c in df.columns), None)
    txc = next((c for c in TXT_CANDS if c in df.columns), None)
    if txc is None:
        objs = [c for c in df.columns if df[c].dtype == object]
        txc = max(objs, key=lambda c: df[c].astype(str).str.len().mean())
    if idc is None:
        print(f"WARNING: no id column in {src}; using row index")
        rid = df.index.astype(str)
    else:
        rid = df[idc].astype(str)
    out = pd.DataFrame({"rid": list(rid), "text": df[txc].astype(str).tolist()})
    out = out[out.text.str.strip() != ""].drop_duplicates("rid").reset_index(drop=True)
    print(f"[{topic}] {len(out)} responses from {src} (id={idc}, text={txc})")
    return out


def _config():
    sys.path.insert(0, str(PIPE))
    import config   # read-only import
    return config


def question_text(topic: str) -> str:
    if TOPICS[topic] == "canada":
        return "\n".join(_config().TOPIC_RELEVANCE_ANCHORS[topic][:3])
    return json.loads((REV / "scotland" / "datasets" / topic / "questions.json").read_text())["question"]


def official_sentences(topic: str):
    if TOPICS[topic] == "canada":
        return split_sentences(_config().TOPICS[topic]["summary"])
    return json.loads((REV / "scotland" / "datasets" / topic / "summary.json").read_text())["sentences"]


def wider_sentences(topic: str):
    """Scotland only: wider full-findings text, for the relocation check."""
    if TOPICS[topic] != "scotland":
        return None
    p = REV / "scotland" / "datasets" / topic / "sensitivity_targets.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("full_findings")


def load_summaries(topic: str, which: str = "official") -> dict:
    out = {"official": official_sentences(topic)}
    if which == "all":
        root = REV / "llm_summaries" / "outputs" / topic
        for p in sorted(root.glob("*/seed_*/summary.json")):
            d = json.loads(p.read_text())
            sents = d.get("sentences") or split_sentences(d.get("text", ""))
            if sents:
                out[f"{p.parent.parent.name}/{p.parent.name}"] = sents
    return out


def other_country_summary(topic: str):
    other = "q7" if TOPICS[topic] == "canada" else "education"
    return other, official_sentences(other)


def tag_for(threshold: float) -> str:
    return f"t{threshold:.2f}"


def topic_dir(topic: str) -> Path:
    d = WORK / topic
    d.mkdir(parents=True, exist_ok=True)
    return d


def add_common_args(ap):
    ap.add_argument("--topics", nargs="+", default=list(TOPICS), choices=list(TOPICS))
    ap.add_argument("--threshold", type=float, default=0.30, help="cosine-distance threshold for idea clustering (a priori 0.30)")
    ap.add_argument("--workers", type=int, default=8)
    return ap
