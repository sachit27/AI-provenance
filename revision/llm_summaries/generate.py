#!/usr/bin/env python3
"""Generate controlled map-reduce summaries. See PROTOCOL.md / README.md.

Run from canadian_data/:
  revision/.venv/bin/python revision/llm_summaries/generate.py --model gpt-4o-mini --topics q7 --seeds 1729
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

OPENAI_PRICE_IN, OPENAI_PRICE_OUT = 0.15, 0.60   # USD per 1M tokens (ASSUMED list prices; verify)
MAP_WORDS, MERGE_WORDS = 250, 400
MAX_RETRIES = 3
CONDENSE_RETRIES = 3
FINAL_FIT_TOKENS = CHUNK_TOKENS        # notes must fit this before the final prompt
GROUP_TOKENS = CHUNK_TOKENS // 2       # merge-group size

MAP_SYSTEM = "You are a careful policy analyst working for a government department."


def est_tokens(s: str) -> int:
    return (len(s) + 3) // 4


# ---------------------------------------------------------------- cache + LLM
class Tally:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.hits = self.tin = self.tout = 0
        self.secs = 0.0

    def add(self, hit, tin, tout, secs):
        with self.lock:
            if hit:
                self.hits += 1
            else:
                self.calls += 1
                self.tin += tin
                self.tout += tout
                self.secs += secs

    def line(self, cfg):
        s = (f"calls={self.calls} cache_hits={self.hits} in_tok={self.tin:,} out_tok={self.tout:,} "
             f"llm_time={self.secs/60:.1f}min")
        if cfg["kind"] == "openai":
            cost = self.tin / 1e6 * OPENAI_PRICE_IN + self.tout / 1e6 * OPENAI_PRICE_OUT
            s += f" est_cost=${cost:.3f} (assumed $0.15/M in, $0.60/M out)"
        return s


class LLM:
    def __init__(self, cfg, host):
        self.cfg, self.host, self.tally = cfg, host.rstrip("/"), Tally()
        self.digest = None
        self.last_resp_model = None
        self.db = sqlite3.connect(CACHE_DB, timeout=120, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self.dblock = threading.Lock()
        if cfg["kind"] == "openai":
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env")
            self.client = OpenAI(max_retries=8, timeout=300)
        else:
            import requests
            self.requests = requests
            tags = requests.get(self.host + "/api/tags", timeout=30).json()["models"]
            found = next((m for m in tags if m["name"] == cfg["name"] or m.get("model") == cfg["name"]), None)
            if not found:
                raise SystemExit(f"Ollama model {cfg['name']!r} not found; have {[m['name'] for m in tags]}")
            self.digest = found["digest"]

    def options(self, seed, max_tokens):
        if self.cfg["kind"] == "openai":
            return {"temperature": TEMPERATURE, "seed": seed, "max_tokens": max_tokens}
        return {"temperature": TEMPERATURE, "seed": seed, "num_ctx": NUM_CTX, "num_predict": max_tokens}

    def chat(self, messages, seed, max_tokens):
        opts = self.options(seed, max_tokens)
        key = sha256([self.cfg["name"], messages, seed, opts, "think=false"])
        with self.dblock:
            row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            self.tally.add(True, 0, 0, 0)
            self._note(v)
            return v["text"]
        t0 = time.time()
        if self.cfg["kind"] == "openai":
            r = self.client.chat.completions.create(model=self.cfg["name"], messages=messages, **opts)
            v = dict(text=r.choices[0].message.content or "", resp_model=r.model,
                     tin=r.usage.prompt_tokens, tout=r.usage.completion_tokens)
        else:
            payload = dict(model=self.cfg["name"], messages=messages, stream=False, think=False,
                           options=opts, keep_alive="30m")
            for i in range(3):
                try:
                    r = self.requests.post(self.host + "/api/chat", json=payload, timeout=7200)
                    r.raise_for_status()
                    break
                except Exception as e:
                    if i == 2:
                        raise
                    print(f"   ollama error {e!r}; retrying", flush=True)
                    time.sleep(5 * (i + 1))
            j = r.json()
            v = dict(text=j["message"]["content"], resp_model=self.cfg["name"],
                     tin=j.get("prompt_eval_count", 0), tout=j.get("eval_count", 0))
        v["secs"] = time.time() - t0
        with self.dblock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(v)))
            self.db.commit()
        self.tally.add(False, v["tin"], v["tout"], v["secs"])
        self._note(v)
        return v["text"]

    def _note(self, v):
        if self.cfg["kind"] == "openai":
            self.last_resp_model = v.get("resp_model")


def clean_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)
    return text.strip()


def one_para(text: str) -> str:
    text = clean_output(text).replace("**", "")
    return " ".join(text.split())


# ------------------------------------------------------------------- protocol
def make_chunks(texts, order, limit_tokens):
    maxc = limit_tokens * 4 - 400
    chunks, cur, cur_c, truncated = [], [], 0, 0
    for i in order:
        t = " ".join(texts[i].split())
        if len(t) > maxc:
            t, truncated = t[:maxc], truncated + 1
        need = len(t) + 12
        if cur and cur_c + need > limit_tokens * 4:
            chunks.append(cur)
            cur, cur_c = [], 0
        cur.append((i, t))
        cur_c += need
    if cur:
        chunks.append(cur)
    return chunks, truncated


def chunk_text(chunk):
    return "\n\n".join(f"[{k+1}] {t}" for k, (_, t) in enumerate(chunk))


def map_messages(question, chunk):
    return [{"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content":
             f"You are analysing public consultation responses to the question: {question}\n\n"
             f"Write concise analytic notes (max {MAP_WORDS} words) on the views expressed in these responses.\n\n"
             f"Responses:\n{chunk_text(chunk)}"}]


def merge_messages(question, notes):
    body = "\n\n".join(f"Notes {k+1}:\n{n}" for k, n in enumerate(notes))
    return [{"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content":
             f"Below are analytic notes on different batches of public consultation responses to the question: {question}\n\n"
             f"Merge them into a single set of concise analytic notes (max {MERGE_WORDS} words) on the views expressed "
             f"across all batches.\n\n{body}"}]


def final_messages(question, notes, n_sent, n_words):
    body = "\n\n".join(f"Notes {k+1}:\n{n}" for k, n in enumerate(notes))
    return [{"role": "system", "content": MAP_SYSTEM},
            {"role": "user", "content":
             f"Below are analytic notes on public consultation responses to the question: {question}\n\n"
             f"Write a summary of what respondents said in exactly {n_sent} sentences and at most {n_words} words. "
             f"Use plain prose: no lists, headings or bullet points. Output only the summary.\n\n{body}"}]


def check(text, n_sent, n_words):
    sents = split_summary(text)
    return sents, len(sents), word_count(text)


def violation(n, w, n_sent, n_words):
    return (abs(n - n_sent), max(0, w - n_words))


def run_one(llm, data, seed, args):
    topic, cfg = data["topic"], llm.cfg
    outdir = OUT_DIR / topic / cfg["slug"] / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    n_sent, n_words, question = data["budget_sentences"], data["budget_words"], data["question"]
    order = list(range(len(data["texts"])))
    random.Random(seed).shuffle(order)
    chunks, truncated = make_chunks(data["texts"], order, args.chunk_tokens)
    input_hash = sha256([question, n_sent, n_words, PROTOCOL_VERSION, args.chunk_tokens,
                         [[data["ids"][i] for i, _ in c] for c in chunks],
                         sha256("\n".join(data["texts"]))])
    sfile = outdir / "summary.json"
    if sfile.exists() and not args.force:
        old = json.loads(sfile.read_text())
        if old.get("input_hash") == input_hash and old.get("protocol_version") == PROTOCOL_VERSION:
            print(f"[{topic}/{cfg['slug']}/seed {seed}] already done; skipping", flush=True)
            return
    print(f"[{topic}/{cfg['slug']}/seed {seed}] {len(data['texts'])} responses -> {len(chunks)} chunks "
          f"(~{sum(est_tokens(chunk_text(c)) for c in chunks):,} tok), budget {n_sent} sent / {n_words} words"
          + (f", {truncated} oversize responses truncated" if truncated else ""), flush=True)
    if args.dry_run:
        return
    t0 = time.time()
    done = [0]

    def do_map(c):
        out = one_para(llm.chat(map_messages(question, c), seed, 700))
        done[0] += 1
        print(f"   map {done[0]}/{len(chunks)}  [{time.time()-t0:.0f}s]  {llm.tally.line(cfg)}", flush=True)
        return out

    if args.workers > 1:
        with ThreadPoolExecutor(args.workers) as ex:
            notes = list(ex.map(do_map, chunks))
    else:
        notes = [do_map(c) for c in chunks]
    (outdir / "map_notes.json").write_text(json.dumps(
        [{"chunk": k, "respondent_ids": [data["ids"][i] for i, _ in c], "notes": n}
         for k, (c, n) in enumerate(zip(chunks, notes))], indent=1))

    levels = []
    level = 0
    while sum(est_tokens(n) for n in notes) > FINAL_FIT_TOKENS and len(notes) > 1:
        groups, cur, cur_t = [], [], 0
        for n in notes:
            if cur and (cur_t + est_tokens(n) > GROUP_TOKENS):
                groups.append(cur); cur, cur_t = [], 0
            cur.append(n); cur_t += est_tokens(n)
        if cur:
            groups.append(cur)
        if len(groups) == len(notes):       # guarantee progress
            groups = [notes[i:i + 2] for i in range(0, len(notes), 2)]
        notes = [one_para(llm.chat(merge_messages(question, g), seed, 900)) for g in groups]
        level += 1
        levels.append(notes)
        print(f"   reduce level {level}: {len(notes)} note sets", flush=True)
    (outdir / "reduce_notes.json").write_text(json.dumps({"levels": levels, "final_notes": notes}, indent=1))

    msgs = final_messages(question, notes, n_sent, n_words)
    attempts = []
    best = None
    max_tok = max(500, 3 * n_words)
    for a in range(1 + MAX_RETRIES):
        text = one_para(llm.chat(msgs, seed, max_tok))
        sents, n, w = check(text, n_sent, n_words)
        ok = (n == n_sent and w <= n_words)
        attempts.append({"attempt": a + 1, "n_sentences": n, "n_words": w, "ok": ok, "text": text})
        v = violation(n, w, n_sent, n_words)
        if best is None or v < best[0]:
            best = (v, text, sents, n, w)
        print(f"   final attempt {a+1}: {n} sentences, {w} words -> {'OK' if ok else 'violates budget'}", flush=True)
        if ok:
            break
        msgs = msgs + [{"role": "assistant", "content": text},
                       {"role": "user", "content":
                        f"That summary had {n} sentences and {w} words. It must have exactly {n_sent} sentences "
                        f"and at most {n_words} words. Rewrite it accordingly, in plain prose, and output only the summary."}]
    repair_used = False
    if not (best[3] == n_sent and best[4] <= n_words):
        # Identical for every model, used only after the chained retries fail: a fresh
        # condensation request on the best attempt (no new content allowed).
        target = max(n_sent * 8, n_words - 10)
        for r in range(CONDENSE_RETRIES):
            cmsgs = [{"role": "system", "content": MAP_SYSTEM},
                     {"role": "user", "content":
                      f"The following summary has {best[3]} sentences and {best[4]} words. Rewrite it so that it has "
                      f"exactly {n_sent} sentences and about {target} words (never more than {n_words} words), by merging, "
                      f"splitting or shortening its sentences. Keep its content and wording where possible; do not add "
                      f"information. Plain prose only. Output only the summary.\n\n{best[1]}"}]
            text = one_para(llm.chat(cmsgs, seed + 1000 * (r + 1), max_tok))
            sents, n, w = check(text, n_sent, n_words)
            ok = (n == n_sent and w <= n_words)
            attempts.append({"attempt": len(attempts) + 1, "kind": "condense", "n_sentences": n,
                             "n_words": w, "ok": ok, "text": text})
            print(f"   condense attempt {r+1}: {n} sentences, {w} words -> {'OK' if ok else 'violates budget'}", flush=True)
            v = violation(n, w, n_sent, n_words)
            if v < best[0]:
                best = (v, text, sents, n, w)
                repair_used = True
            if ok:
                break
    _, text, sents, n, w = best
    (outdir / "final_attempts.json").write_text(json.dumps(attempts, indent=1))
    digest = llm.digest if cfg["kind"] == "ollama" else f"openai:{llm.last_resp_model}"
    summary = dict(
        topic=topic, model=cfg["name"], model_slug=cfg["slug"], model_digest=digest, seed=seed,
        sentences=sents, text=text, n_sentences=n, n_words=w,
        budget_sentences=n_sent, budget_words=n_words,
        budget_compliant=bool(n == n_sent and w <= n_words), attempts=len(attempts), condense_repair_used=repair_used,
        protocol_version=PROTOCOL_VERSION, input_hash=input_hash,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    sfile.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[{topic}/{cfg['slug']}/seed {seed}] done in {(time.time()-t0)/60:.1f} min; compliant={summary['budget_compliant']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--topics", nargs="+", default=ALL_TOPICS, choices=ALL_TOPICS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--workers", type=int, default=None, help="parallel map calls (default 4 for OpenAI, 1 for Ollama)")
    ap.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    ap.add_argument("--id-col"); ap.add_argument("--text-col")
    ap.add_argument("--allow-records-fallback", action="store_true",
                    help="Scotland only: read revision/scotland/datasets/<q>/records.jsonl if clean.parquet is missing")
    ap.add_argument("--force", action="store_true", help="rebuild summary.json even if present (LLM calls still cached)")
    ap.add_argument("--dry-run", action="store_true", help="only report chunking; no LLM calls")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    if args.workers is None:
        args.workers = 4 if cfg["kind"] == "openai" else 1
    llm = None if args.dry_run else LLM(cfg, args.host)
    if args.dry_run:
        llm = type("D", (), {"cfg": cfg})()
    for topic in args.topics:
        try:
            data = load_topic(topic, args.id_col, args.text_col, args.allow_records_fallback)
        except FileNotFoundError as e:
            print(f"SKIP {topic}: {e}", flush=True)
            continue
        for seed in args.seeds:
            run_one(llm, data, seed, args)
    if not args.dry_run:
        print("TOTAL", cfg["name"], llm.tally.line(cfg))


if __name__ == "__main__":
    main()
